package io.github.lbenedetto.internal.floattensor

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorShape
import jdk.incubator.vector.VectorSpecies
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import kotlin.math.exp

internal abstract class FloatTensor {
  abstract val size: Long

  abstract fun getFloat(index: Long): Float

  abstract fun setFloat(index: Int, value: Float)

  abstract fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector?

  open fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
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
    return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Math::max)
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
    val VECTOR_BIT_SIZE: Int = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize())
    val USE_VECTOR_API: Boolean = VECTOR_BIT_SIZE != 0

    val F_SPECIES: VectorSpecies<Float>?
    val I_SPECIES: VectorSpecies<Int>?
    val S_SPECIES_HALF: VectorSpecies<Short>?

    init {
      if (USE_VECTOR_API) {
        F_SPECIES =
          VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(Float::class.javaPrimitiveType)
        I_SPECIES = F_SPECIES.withLanes(Int::class.javaPrimitiveType)
        S_SPECIES_HALF =
          VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(Short::class.javaPrimitiveType)
        assert(F_SPECIES.length() == S_SPECIES_HALF.length())
      } else {
        F_SPECIES = null
        I_SPECIES = null
        S_SPECIES_HALF = null
      }
    }

    fun Byte.toUnsignedInt(): Int = toInt() and 0xFF

    fun Short.toUnsignedInt(): Int = toInt() and 0xFFFF

    fun readShort(memorySegment: MemorySegment, offset: Long): Short {
      return memorySegment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset)
    }

    fun readFloat16(memorySegment: MemorySegment, offset: Long): Float {
      return java.lang.Float.float16ToFloat(readShort(memorySegment, offset))
    }

    fun readByte(memorySegment: MemorySegment, offset: Long): Byte {
      return memorySegment.get(ValueLayout.JAVA_BYTE, offset)
    }

    fun readFloat(memorySegment: MemorySegment, offset: Long): Float {
      return memorySegment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset)
    }

    fun numberOfElements(vararg dimensions: Int): Int {
      assert(dimensions.all { it > 0 })
      return dimensions.reduce(Math::multiplyExact)
    }

    fun numberOfElementsLong(vararg dimensions: Int): Long {
      var result: Long = 1
      for (d in dimensions) {
        assert(d > 0)
        result = Math.multiplyExact(result, d)
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
