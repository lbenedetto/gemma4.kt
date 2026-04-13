package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.data.MemorySegment
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.util.assert
import io.github.lbenedetto.internal.util.vectorMathEnabled

internal class Q4_0FloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloat(index: Long): Float {
    assert(index in 0..<size)
    val blockIndex = index / GGMLType.Q4_0.blockSize
    val blockOffset = blockIndex * GGMLType.Q4_0.typeSize
    val scale: Float = memorySegment.readFloat16(blockOffset)
    var quant: Byte
    val modIndex = (index % GGMLType.Q4_0.blockSize).toInt()
    quant = if (modIndex < GGMLType.Q4_0.blockSize / 2) {
      val offset = blockOffset + Float16.BYTES + modIndex
      (memorySegment.readByte(offset).toInt() and 0x0F).toByte()
    } else {
      val offset = blockOffset + Float16.BYTES + modIndex - GGMLType.Q4_0.blockSize / 2
      ((memorySegment.readByte(offset).toInt() ushr 4) and 0x0F).toByte()
    }
    quant = (quant - 8).toByte()
    return quant * scale
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (vectorMathEnabled()) {
      Q4_0FloatTensorMath.vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }
}
