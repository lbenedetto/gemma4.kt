package com.llama4j.floattensor

import com.llama4j.gguf.GGMLType
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

  public override fun type(): GGMLType {
    return GGMLType.F32
  }

  override fun fillInPlace(thisOffset: Int, size: Int, value: Float): FloatTensor {
    Arrays.fill(values, thisOffset, thisOffset + size, value)
    return this
  }

  public override fun getFloatVector(species: VectorSpecies<Float>, index: Int): FloatVector {
    if (!FloatTensor.Companion.USE_VECTOR_API) {
      throw UnsupportedOperationException()
    }
    return FloatVector.fromArray(species, values, index)
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor) {
      if (FloatTensor.Companion.USE_VECTOR_API) {
        return vectorDot(this, thisOffset, that, thatOffset, size)
      }
      return FloatTensor.Companion.scalarDot(this, thisOffset, that, thatOffset, size)
    }
    return that.dot(thatOffset, this, thisOffset, size)
  }

  companion object {
    fun allocate(vararg dims: Int): FloatTensor {
      val numberOfElements: Int = FloatTensor.Companion.numberOfElements(*dims)
      return ArrayFloatTensor(FloatArray(numberOfElements))
    }

    private fun vectorDot(
      thiz: ArrayFloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      var `val` = FloatVector.zero(Objects.requireNonNull<VectorSpecies<Float>?>(FloatTensor.Companion.F_SPECIES))
      val upperBound: Int = FloatTensor.Companion.F_SPECIES.loopBound(size)
      run {
        var i = 0
        while (i < upperBound) {
          val a = FloatVector.fromArray(FloatTensor.Companion.F_SPECIES, thiz.values, thisOffset + i)
          val b = FloatVector.fromArray(FloatTensor.Companion.F_SPECIES, that.values, thatOffset + i)
          `val` = a.fma(b, `val`)
          i += FloatTensor.Companion.F_SPECIES.length()
        }
      }
      var result = `val`.reduceLanes(VectorOperators.ADD)
      for (i in upperBound..<size) {
        result += thiz.values[thisOffset + i] * that.values[thatOffset + i]
      }
      return result
    }
  }
}
