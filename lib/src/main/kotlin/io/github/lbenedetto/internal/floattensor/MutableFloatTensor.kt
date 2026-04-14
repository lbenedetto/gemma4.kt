package io.github.lbenedetto.internal.floattensor

import kotlin.math.exp

internal interface MutableFloatTensor : FloatTensor {
  fun setFloat(index: Int, value: Float)

  fun mapInPlace(thisOffset: Int, size: Int, mapFunction: (Float) -> Float): FloatTensor {
    val endIndex = thisOffset + size
    for (i in thisOffset..<endIndex) {
      setFloat(i, mapFunction(this[i]))
    }
    return this
  }

  fun mapInPlace(mapFunction: (Float) -> Float): FloatTensor {
    return mapInPlace(0, Math.toIntExact(size), mapFunction)
  }

  fun mapWithIndexInPlace(thisOffset: Int, size: Int, mapWithIndexFunction: (Float, Int) -> Float): FloatTensor {
    val endOffset = thisOffset + size
    for (i in thisOffset..<endOffset) {
      setFloat(i, mapWithIndexFunction(this[i], i))
    }
    return this
  }

  fun addInPlace(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): FloatTensor {
    return mapWithIndexInPlace(
      thisOffset,
      size
    ) { value, index -> value + that[index - thisOffset + thatOffset] }
  }

  fun addInPlace(that: FloatTensor): FloatTensor {
    return addInPlace(0, that, 0, Math.toIntExact(size))
  }

  fun multiplyInPlace(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): FloatTensor {
    return mapWithIndexInPlace(
      thisOffset,
      size
    ) { value, index -> value * that[index - thisOffset + thatOffset] }
  }

  fun divideInPlace(thisOffset: Int, size: Int, value: Float): FloatTensor {
    return mapInPlace(thisOffset, size) { it / value }
  }

  fun fillInPlace(thisOffset: Int, size: Int, value: Float): FloatTensor {
    return mapInPlace(thisOffset, size) { value }
  }

  fun softmaxInPlace(thisOffset: Int, size: Int): FloatTensor {
    val maxVal = max(thisOffset, size)
    mapInPlace(thisOffset, size) { exp((it - maxVal).toDouble()).toFloat() }
    val sum = sum(thisOffset, size)
    return divideInPlace(thisOffset, size, sum)
  }

  fun saxpyInPlace(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int, a: Float): FloatTensor {
    for (i in 0..<size) {
      setFloat(thisOffset + i, a * that[thatOffset + i] + this[thisOffset + i])
    }
    return this
  }
}
