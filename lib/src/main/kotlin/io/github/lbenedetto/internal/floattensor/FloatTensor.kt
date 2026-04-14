package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.scalarDot
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies

internal interface FloatTensor {
  val size: Long

  fun getFloat(index: Long): Float

  fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector?

  fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return scalarDot(this, thisOffset, that, thatOffset, size)
  }

  fun matmul(that: FloatTensor, out: MutableFloatTensor, dim0: Int, dim1: Int) {
    parallelFor(0, dim0) { out.setFloat(it, dot(it * dim1, that, 0, dim1)) }
  }

  // matmul with offset into this tensor (for expert weight slicing in 3D tensors)
  fun matmul(that: FloatTensor, out: MutableFloatTensor, dim0: Int, dim1: Int, thisOffset: Int) {
    parallelFor(0, dim0) { out.setFloat(it, dot(thisOffset + it * dim1, that, 0, dim1)) }
  }

  fun reduce(thisOffset: Int, size: Int, seed: Float, reduce: (Float, Float) -> Float): Float {
    var result = seed
    for (i in 0..<size) {
      result = reduce(result, getFloat((thisOffset + i).toLong()))
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
    that.mapWithIndexInPlace(
      thatOffset,
      size
    ) { _, index -> this.getFloat((index - thatOffset + thisOffset).toLong()) }
  }

  fun argmax(thisOffset: Int, size: Int): Int {
    assert(size > 0)
    var maxIndex = thisOffset
    var maxValue = this.getFloat(maxIndex.toLong())
    val endIndex = thisOffset + size
    for (i in thisOffset..<endIndex) {
      val f = this.getFloat(i.toLong())
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
