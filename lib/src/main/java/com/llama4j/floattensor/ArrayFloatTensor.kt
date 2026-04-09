package com.llama4j.floattensor

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.nio.FloatBuffer
import java.util.*

class ArrayFloatTensor : FloatTensor {
  val size: Long
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

  override fun size(): Long {
    return size
  }

  override fun getFloat(index: Long): Float {
    return values[Math.toIntExact(index)]
  }

  override fun setFloat(index: Int, value: Float) {
    values[index] = value
  }

  override fun type(): com.llama4j.gguf.GGMLType {
    return _root_ide_package_.com.llama4j.gguf.GGMLType.F32
  }

  override fun fillInPlace(thisOffset: Int, size: Int, value: Float): FloatTensor {
    Arrays.fill(values, thisOffset, thisOffset + size, value)
    return this
  }

  override fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector {
    if (!USE_VECTOR_API) {
      throw UnsupportedOperationException()
    }
    return FloatVector.fromArray(species, values, offset)
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor) {
      if (USE_VECTOR_API) {
        return vectorDot(this, thisOffset, that, thatOffset, size)
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

    private fun vectorDot(
      thiz: ArrayFloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      var value = FloatVector.zero(F_SPECIES!!)
      val upperBound: Int = F_SPECIES.loopBound(size)
      run {
        var i = 0
        while (i < upperBound) {
          val a = FloatVector.fromArray(F_SPECIES, thiz.values, thisOffset + i)
          val b = FloatVector.fromArray(F_SPECIES, that.values, thatOffset + i)
          value = a.fma(b, value)
          i += F_SPECIES.length()
        }
      }
      var result = value.reduceLanes(VectorOperators.ADD)
      for (i in upperBound..<size) {
        result += thiz.values[thisOffset + i] * that.values[thatOffset + i]
      }
      return result
    }
  }
}
