package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.data.FloatBuffer
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.numberOfElements
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.util.Math
import io.github.lbenedetto.internal.util.vectorMathEnabled

internal class ArrayFloatTensor : FloatTensor {
  override val size: Long
  val values: FloatArray

  internal constructor(values: FloatArray) {
    this.size = values.size.toLong()
    this.values = values
  }

  internal constructor(buf: FloatBuffer) {
    this.values = FloatArray(buf.remaining())
    this.size = values.size.toLong()
    buf.get(this.values)
    buf.rewind()
  }

  override fun getFloat(index: Long): Float {
    return values[Math.toIntExact(index)]
  }

  override fun setFloat(index: Int, value: Float) {
    values[index] = value
  }

  override fun fillInPlace(thisOffset: Int, size: Int, value: Float): FloatTensor {
    values.fill(value, thisOffset, thisOffset + size)
    return this
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor) {
      if (vectorMathEnabled()) {
        return ArrayFloatTensorMath.vectorDot(this, thisOffset, that, thatOffset, size)
      }
      return scalarDot(this, thisOffset, that, thatOffset, size)
    }
    return that.dot(thatOffset, this, thisOffset, size)
  }

  companion object {
    fun allocate(vararg dims: Int): FloatTensor {
      val numberOfElements: Int = numberOfElements(*dims)
      return ArrayFloatTensor(FloatArray(numberOfElements))
    }
  }
}
