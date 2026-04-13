package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.util.vectorMathEnabled
import java.lang.foreign.MemorySegment

internal class Q8_0FloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloat(index: Long): Float {
    val blockIndex = index / GGMLType.Q8_0.blockSize
    val withinBlockIndex = index % GGMLType.Q8_0.blockSize
    val blockOffset = blockIndex * GGMLType.Q8_0.typeSize
    val quant: Byte = memorySegment.readByte(blockOffset + Float16.BYTES + withinBlockIndex)
    val scale: Float = memorySegment.readFloat16(blockOffset)
    return quant * scale
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (vectorMathEnabled()) {
      Q8_0FloatTensorMath.vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }
}
