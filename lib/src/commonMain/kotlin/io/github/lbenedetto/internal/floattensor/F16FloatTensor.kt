package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.data.MemorySegment
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.util.assert
import io.github.lbenedetto.internal.util.vectorMathEnabled

internal class F16FloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloat(index: Long): Float {
    assert(index in 0..<size)
    return memorySegment.readFloat16(index * 2)
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (vectorMathEnabled()) {
      F16FloatTensorMath.vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }
}
