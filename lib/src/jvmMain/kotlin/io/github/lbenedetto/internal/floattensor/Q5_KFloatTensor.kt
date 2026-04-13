package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.data.MemorySegment
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.toUnsignedInt
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.gguf.QK_K
import io.github.lbenedetto.internal.util.vectorMathEnabled

internal class Q5_KFloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloat(index: Long): Float {
    val blockIndex: Long = index / BLOCK_SIZE
    val withinBlock = (index % BLOCK_SIZE).toInt()
    val blockOffset: Long = blockIndex * TYPE_SIZE
    val d: Float = memorySegment.readFloat16(blockOffset)
    val dmin: Float = memorySegment.readFloat16(blockOffset + 2)
    val scalesOffset = blockOffset + 4
    val qhOffset = blockOffset + 16 // 4 + 12
    val qsOffset = blockOffset + 48 // 4 + 12 + 32

    val group = withinBlock / 64
    val inGroup = withinBlock % 64
    val isHigh = inGroup >= 32
    val l = if (isHigh) inGroup - 32 else inGroup
    val subBlock = if (isHigh) group * 2 + 1 else group * 2

    val sc: Int = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, false)
    val m: Int = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, true)

    val qsByte: Byte = memorySegment.readByte(qsOffset + group * 32 + l)
    val nibble = if (isHigh) ((qsByte.toUnsignedInt() shr 4) and 0xF) else (qsByte.toUnsignedInt() and 0xF)

    val qhBitPos = if (isHigh) 2 * group + 1 else 2 * group
    val qhBit = (memorySegment.readByte(qhOffset + l).toUnsignedInt() shr qhBitPos) and 1

    val quant = nibble or (qhBit shl 4)
    return d * sc * quant - dmin * m
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (vectorMathEnabled()) {
      Q5_KFloatTensorMath.vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }

  companion object {
    const val BLOCK_SIZE: Int = QK_K
    val TYPE_SIZE: Int = GGMLType.Q5_K.typeSize
  }
}
