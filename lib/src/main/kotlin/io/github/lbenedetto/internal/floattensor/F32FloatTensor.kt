package io.github.lbenedetto.internal.floattensor

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.ByteOrder

internal class F32FloatTensor(
  override val size: Long,
  private val memorySegment: MemorySegment
) : FloatTensor() {

  override fun getFloat(index: Long): Float {
    return memorySegment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, index * Float.SIZE_BYTES)
  }

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("read-only")
  }

  override fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector {
    if (!USE_VECTOR_API) {
      throw UnsupportedOperationException()
    }
    return FloatVector.fromMemorySegment(species, memorySegment, offset.toLong() * Float.SIZE_BYTES, ByteOrder.LITTLE_ENDIAN)
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor && USE_VECTOR_API) {
      return vectorDot(this, thisOffset, that, thatOffset, size)
    }
    return scalarDot(this, thisOffset, that, thatOffset, size)
  }

  companion object {
    private fun vectorDot(
      thiz: F32FloatTensor,
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
          val a = FloatVector.fromMemorySegment(
            F_SPECIES,
            thiz.memorySegment,
            (thisOffset + i).toLong() * Float.SIZE_BYTES,
            ByteOrder.LITTLE_ENDIAN
          )
          val b = FloatVector.fromArray(F_SPECIES, that.values, thatOffset + i)
          value = a.fma(b, value)
          i += F_SPECIES.length()
        }
      }
      var result = value.reduceLanes(VectorOperators.ADD)
      for (i in upperBound..<size) {
        result += thiz.getFloat((thisOffset + i).toLong()) * that.values[thatOffset + i]
      }
      return result
    }
  }
}
