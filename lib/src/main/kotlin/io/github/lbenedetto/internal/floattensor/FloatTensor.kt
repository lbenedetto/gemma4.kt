package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.scalarDot

internal interface FloatTensor {
  val size: Long

  operator fun get(index: Long): Float

  operator fun get(index: Int): Float {
    return get(index.toLong())
  }

  fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return scalarDot(this, thisOffset, that, thatOffset, size)
  }

  fun matmul(that: FloatTensor, out: MutableFloatTensor, dim0: Int, dim1: Int) {
    parallelFor(0, dim0) {
      out.setFloat(it, dot(it * dim1, that, 0, dim1))
    }
  }

  // matmul with offset into this tensor (for expert weight slicing in 3D tensors)
  fun matmul(that: FloatTensor, out: MutableFloatTensor, dim0: Int, dim1: Int, thisOffset: Int) {
    parallelFor(0, dim0) {
      out.setFloat(it, dot(thisOffset + it * dim1, that, 0, dim1))
    }
  }

  fun reduce(thisOffset: Int, size: Int, seed: Float, reduce: (Float, Float) -> Float): Float {
    var result = seed
    for (i in 0..<size) {
      result = reduce(result, this[thisOffset + i])
    }
    return result
  }

  fun sum(thisOffset: Int, size: Int): Float {
    return reduce(thisOffset, size, 0f, Float::plus)
  }

  fun max(thisOffset: Int, size: Int): Float {
    return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Math::max)
  }

  fun copyTo(thisOffset: Int, that: MutableFloatTensor, thatOffset: Int, size: Int) {
    that.mapWithIndexInPlace(thatOffset, size) { _, index ->
      this[index - thatOffset + thisOffset]
    }
  }

  fun argmax(thisOffset: Int, size: Int): Int {
    assert(size > 0)
    var maxIndex = thisOffset
    var maxValue = this[maxIndex]
    val endIndex = thisOffset + size
    for (i in thisOffset..<endIndex) {
      val f = this[i]
      if (f > maxValue) {
        maxValue = f
        maxIndex = i
      }
    }
    return maxIndex
  }

  fun argmax(): Int {
    return argmax(0, Math.toIntExact(size))
  }
}
