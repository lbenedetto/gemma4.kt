package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.data.MemorySegment
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.toUnsignedInt
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.gguf.QK_K
import io.github.lbenedetto.internal.util.vectorMathEnabled

internal class Q6_KFloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  // Block layout: ql[128] | qh[64] | scales[16] (int8) | d (fp16)
  // 256 elements per block, 6-bit quants: 4 from ql nibble + 2 from qh
  override fun getFloat(index: Long): Float {
    val blockIndex: Long = index / BLOCK_SIZE
    val withinBlock = (index % BLOCK_SIZE).toInt()
    val blockOffset: Long = blockIndex * TYPE_SIZE
    val qhOff = blockOffset + 128
    val scOff = blockOffset + 192
    val d: Float = memorySegment.readFloat16(blockOffset + 208)

    val half = withinBlock / 128
    val rem128 = withinBlock % 128
    val sub32 = rem128 / 32
    val l = rem128 % 32

    val qlBase = blockOffset + half * 64
    val qhBase = qhOff + half * 32

    val qlNibble: Int
    val qhShift: Int
    when (sub32) {
      0 -> {
        qlNibble = memorySegment.readByte(qlBase + l).toUnsignedInt() and 0xF
        qhShift = 0
      }

      1 -> {
        qlNibble = memorySegment.readByte(qlBase + 32 + l).toUnsignedInt() and 0xF
        qhShift = 2
      }

      2 -> {
        qlNibble = (memorySegment.readByte(qlBase + l).toUnsignedInt() shr 4) and 0xF
        qhShift = 4
      }

      3 -> {
        qlNibble = (memorySegment.readByte(qlBase + 32 + l).toUnsignedInt() shr 4) and 0xF
        qhShift = 6
      }

      else -> throw IllegalStateException()
    }

    val qhBits = (memorySegment.readByte(qhBase + l).toUnsignedInt() shr qhShift) and 3
    val q6 = (qlNibble or (qhBits shl 4)) - 32
    val sc = memorySegment.readByte(scOff + half * 8 + sub32 * 2 + l / 16).toInt() // signed int8

    return d * sc * q6
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (vectorMathEnabled()) {
      Q6_KFloatTensorMath.vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }

  companion object {
    const val BLOCK_SIZE: Int = QK_K
    val TYPE_SIZE: Int = GGMLType.Q6_K.typeSize

    internal fun fp16ToFloatNoIntrinsic(h: Short): Float {
      val bits = h.toUnsignedInt()
      val sign = (bits and 0x8000) shl 16
      val exp = (bits ushr 10) and 0x1F
      var mantissa = bits and 0x03FF

      if (exp == 0) {
        if (mantissa == 0) {
          return Float.fromBits(sign)
        }
        var e = 127 - 15 + 1
        while ((mantissa and 0x0400) == 0) {
          mantissa = mantissa shl 1
          e--
        }
        mantissa = mantissa and 0x03FF
        return Float.fromBits(sign or (e shl 23) or (mantissa shl 13))
      }
      if (exp == 0x1F) {
        return Float.fromBits(sign or 0x7F800000 or (mantissa shl 13))
      }
      return Float.fromBits(sign or ((exp + (127 - 15)) shl 23) or (mantissa shl 13))
    }

  }
}
