package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.util.Math
import io.github.lbenedetto.internal.util.assert
import kotlin.math.exp
import kotlin.math.max

interface FloatTensor {
  val size: Long

  fun getFloat(index: Long): Float

  fun setFloat(index: Int, value: Float)

  fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return scalarDot(this, thisOffset, that, thatOffset, size)
  }

  fun matmul(that: FloatTensor, out: FloatTensor, dim0: Int, dim1: Int) {
    parallelFor(0, dim0) { out.setFloat(it, dot(it * dim1, that, 0, dim1)) }
  }

  // matmul with offset into this tensor (for expert weight slicing in 3D tensors)
  fun matmul(that: FloatTensor, out: FloatTensor, dim0: Int, dim1: Int, thisOffset: Int) {
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
    return reduce(thisOffset, size, Float.NEGATIVE_INFINITY) { a, b -> max(a, b) }
  }

  fun copyTo(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int) {
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

  fun mapInPlace(thisOffset: Int, size: Int, mapFunction: (Float) -> Float): FloatTensor {
    val endIndex = thisOffset + size
    for (i in thisOffset..<endIndex) {
      setFloat(i, mapFunction(getFloat(i.toLong())))
    }
    return this
  }

  fun mapInPlace(mapFunction: (Float) -> Float): FloatTensor {
    return mapInPlace(0, Math.toIntExact(size), mapFunction)
  }

  fun mapWithIndexInPlace(thisOffset: Int, size: Int, mapWithIndexFunction: (Float, Int) -> Float): FloatTensor {
    val endOffset = thisOffset + size
    for (i in thisOffset..<endOffset) {
      setFloat(i, mapWithIndexFunction(getFloat(i.toLong()), i))
    }
    return this
  }

  fun addInPlace(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): FloatTensor {
    return mapWithIndexInPlace(
      thisOffset,
      size
    ) { value, index -> value + that.getFloat((index - thisOffset + thatOffset).toLong()) }
  }

  fun addInPlace(that: FloatTensor): FloatTensor {
    return addInPlace(0, that, 0, Math.toIntExact(size))
  }

  fun multiplyInPlace(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): FloatTensor {
    return mapWithIndexInPlace(
      thisOffset,
      size
    ) { value, index -> value * that.getFloat((index - thisOffset + thatOffset).toLong()) }
  }

  fun divideInPlace(thisOffset: Int, size: Int, value: Float): FloatTensor {
    return mapInPlace(thisOffset, size) { it / value }
  }

  open fun fillInPlace(thisOffset: Int, size: Int, value: Float): FloatTensor {
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
      setFloat(thisOffset + i, a * that.getFloat((thatOffset + i).toLong()) + this.getFloat((thisOffset + i).toLong()))
    }
    return this
  }

  companion object {

    fun Byte.toUnsignedInt(): Int = toInt() and 0xFF

    fun Short.toUnsignedInt(): Int = toInt() and 0xFFFF

    fun numberOfElements(vararg dimensions: Int): Int {
      assert(dimensions.all { it > 0 })
      return dimensions.reduce(Math::multiplyExact)
    }

    fun numberOfElementsLong(vararg dimensions: Int): Long {
      var result: Long = 1
      for (d in dimensions) {
        assert(d > 0)
        result = Math.multiplyExact(result, d.toLong())
      }
      return result
    }

    fun scalarDot(thiz: FloatTensor, thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
      var result = 0f
      for (j in 0..<size) {
        result += thiz.getFloat((thisOffset + j).toLong()) * that.getFloat((thatOffset + j).toLong())
      }
      return result
    }
  }
}
