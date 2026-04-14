package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.F_SPECIES
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.USE_VECTOR_API
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.readByte
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.readFloat16
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.readShort
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.scalarDot
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.toUnsignedInt
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.gguf.QK_K
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder
import kotlin.math.min

internal class Q6_KFloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor {

  // Block layout: ql[128] | qh[64] | scales[16] (int8) | d (fp16)
  // 256 elements per block, 6-bit quants: 4 from ql nibble + 2 from qh
  override fun get(index: Long): Float {
    val blockIndex: Long = index / BLOCK_SIZE
    val withinBlock = (index % BLOCK_SIZE).toInt()
    val blockOffset: Long = blockIndex * TYPE_SIZE
    val qhOff = blockOffset + 128
    val scOff = blockOffset + 192
    val d: Float = readFloat16(memorySegment, blockOffset + 208)

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
        qlNibble = readByte(memorySegment, qlBase + l).toUnsignedInt() and 0xF
        qhShift = 0
      }

      1 -> {
        qlNibble = readByte(memorySegment, qlBase + 32 + l).toUnsignedInt() and 0xF
        qhShift = 2
      }

      2 -> {
        qlNibble = (readByte(memorySegment, qlBase + l).toUnsignedInt() shr 4) and 0xF
        qhShift = 4
      }

      3 -> {
        qlNibble = (readByte(memorySegment, qlBase + 32 + l).toUnsignedInt() shr 4) and 0xF
        qhShift = 6
      }

      else -> throw IllegalStateException()
    }

    val qhBits = (readByte(memorySegment, qhBase + l).toUnsignedInt() shr qhShift) and 3
    val q6 = (qlNibble or (qhBits shl 4)) - 32
    val sc = readByte(memorySegment, scOff + half * 8 + sub32 * 2 + l / 16).toInt() // signed int8

    return d * sc * q6
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (USE_VECTOR_API) {
      vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }

  companion object {
    const val BLOCK_SIZE: Int = QK_K
    val TYPE_SIZE: Int = GGMLType.Q6_K.typeSize

    private fun fp16ToFloatNoIntrinsic(h: Short): Float {
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

    private fun vectorDot(
      thiz: Q6_KFloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      var result = 0f
      var j = 0

      assert(Integer.bitCount(BLOCK_SIZE) == 1) { "power of 2" }
      val alignmentBound = min(size, -thisOffset and (BLOCK_SIZE - 1))
      if (alignmentBound > 0) {
        result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
        j += alignmentBound
      }

      var acc = FloatVector.zero(F_SPECIES!!)
      var blockOffset: Long = (thisOffset + j).toLong() / BLOCK_SIZE * TYPE_SIZE
      val upperBound: Int = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE

      while (j < upperBound) {
        val qlOff = blockOffset
        val qhOff = blockOffset + 128
        val scOff = blockOffset + 192
        // NOTE: Deliberately avoid Float.float16ToFloat here.
        // In native-image builds, Graal can lower that intrinsic to VCVTPH2PS with
        // an illegal high XMM operand under heavy vector register pressure in Q6_K
        // vectorDot, causing a compile-time crash. Keep this software conversion
        // until the Graal backend bug is fixed.
        val d: Float =
          fp16ToFloatNoIntrinsic(readShort(thiz.memorySegment, blockOffset + 208))

        for (h in 0..1) {
          val qlBase = qlOff + h * 64
          val qhBase = qhOff + h * 32

          val base = thatOffset + j + h * 128
          for (c in 0..1) {
            val qlA = ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128, thiz.memorySegment,
              qlBase + c * 16L, ByteOrder.LITTLE_ENDIAN
            )
            val qlB = ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128, thiz.memorySegment,
              qlBase + 32 + c * 16L, ByteOrder.LITTLE_ENDIAN
            )
            val qhV = ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128, thiz.memorySegment,
              qhBase + c * 16L, ByteOrder.LITTLE_ENDIAN
            )

            val q0: ByteVector = qlA
              .and(0xF.toByte())
              .or(qhV.and(3.toByte()).lanewise(VectorOperators.LSHL, 4L))
              .sub(32.toByte())
            val q1: ByteVector = qlB
              .and(0xF.toByte())
              .or(qhV.lanewise(VectorOperators.LSHR, 2L).and(3.toByte()).lanewise(VectorOperators.LSHL, 4L))
              .sub(32.toByte())
            val q2: ByteVector = qlA
              .lanewise(VectorOperators.LSHR, 4L)
              .or(qhV.lanewise(VectorOperators.LSHR, 4L).and(3.toByte()).lanewise(VectorOperators.LSHL, 4L))
              .sub(32.toByte())
            val q3: ByteVector = qlB
              .lanewise(VectorOperators.LSHR, 4L)
              .or(qhV.lanewise(VectorOperators.LSHR, 6L).and(3.toByte()).lanewise(VectorOperators.LSHL, 4L))
              .sub(32.toByte())

            val ds0: Float = d * readByte(thiz.memorySegment, scOff + h * 8 + c)
            val ds1: Float = d * readByte(thiz.memorySegment, scOff + h * 8 + 2 + c)
            val ds2: Float = d * readByte(thiz.memorySegment, scOff + h * 8 + 4 + c)
            val ds3: Float = d * readByte(thiz.memorySegment, scOff + h * 8 + 6 + c)

            val ds0Vec = FloatVector.broadcast(F_SPECIES, ds0)
            val ds1Vec = FloatVector.broadcast(F_SPECIES, ds1)
            val ds2Vec = FloatVector.broadcast(F_SPECIES, ds2)
            val ds3Vec = FloatVector.broadcast(F_SPECIES, ds3)

            val sg0Idx = base + c * 16
            val sg1Idx = base + 32 + c * 16
            val sg2Idx = base + 64 + c * 16
            val sg3Idx = base + 96 + c * 16

            when (F_SPECIES.vectorBitSize()) {
              512 -> {
                val q0f = q0.castShape(F_SPECIES, 0).reinterpretAsFloats()
                val q1f = q1.castShape(F_SPECIES, 0).reinterpretAsFloats()
                val q2f = q2.castShape(F_SPECIES, 0).reinterpretAsFloats()
                val q3f = q3.castShape(F_SPECIES, 0).reinterpretAsFloats()
                acc = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx), acc)
                acc = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx), acc)
                acc = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx), acc)
                acc = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx), acc)
              }

              256 -> {
                for (p in 0..1) {
                  val off: Int = p * F_SPECIES.length()
                  val q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats()
                  val q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats()
                  val q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats()
                  val q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats()
                  acc = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), acc)
                  acc = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), acc)
                  acc = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), acc)
                  acc = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), acc)
                }
              }

              128 -> {
                for (p in 0..3) {
                  val off: Int = p * F_SPECIES.length()
                  val q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats()
                  val q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats()
                  val q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats()
                  val q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats()
                  acc = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), acc)
                  acc = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), acc)
                  acc = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), acc)
                  acc = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), acc)
                }
              }

              else -> throw UnsupportedOperationException(F_SPECIES.toString())
            }
          }
        }
        j += BLOCK_SIZE
        blockOffset += TYPE_SIZE.toLong()
      }

      result += acc.reduceLanes(VectorOperators.ADD)

      if (j < size) {
        result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
      }

      return result
    }
  }
}
