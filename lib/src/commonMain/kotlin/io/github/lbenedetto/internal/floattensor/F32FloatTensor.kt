package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.util.MemorySegment

internal class F32FloatTensor(
  override val size: Long,
  internal val memorySegment: MemorySegment
) : FloatTensor {

  override fun getFloat(index: Long): Float {
    return memorySegment.readFloat(index * Float.SIZE_BYTES)
  }

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("read-only")
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor && USE_VECTOR_API) {
      return F32FloatTensorMath.vectorDot(this, thisOffset, that, thatOffset, size)
    }
    return scalarDot(this, thisOffset, that, thatOffset, size)
  }
}
