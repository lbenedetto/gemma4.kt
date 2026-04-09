package com.llama4j.floattensor

import com.llama4j.gguf.GGMLType
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.Float
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.ByteOrder
import java.util.*
import kotlin.Int
import kotlin.Long
import kotlin.UnsupportedOperationException
import kotlin.run

internal class F32FloatTensor(numElements: Long, memorySegment: MemorySegment) : FloatTensor() {
  private val size: Long
  private val memorySegment: MemorySegment

  init {
    this.size = numElements
    this.memorySegment = memorySegment
  }

  override fun size(): Long {
    return size
  }

  override fun getFloat(index: Long): Float {
    return memorySegment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, index * Float.BYTES)
  }

  override fun setFloat(index: Int, value: kotlin.Float) {
    throw UnsupportedOperationException("read-only")
  }

  public override fun type(): GGMLType {
    return GGMLType.F32
  }

  public override fun getFloatVector(species: VectorSpecies<kotlin.Float>, index: Int): FloatVector {
    if (!FloatTensor.Companion.USE_VECTOR_API) {
      throw UnsupportedOperationException()
    }
    return FloatVector.fromMemorySegment(species, memorySegment, index.toLong() * Float.BYTES, ByteOrder.LITTLE_ENDIAN)
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): kotlin.Float {
    if (that is ArrayFloatTensor && FloatTensor.Companion.USE_VECTOR_API) {
      return vectorDot(this, thisOffset, that, thatOffset, size)
    }
    return FloatTensor.Companion.scalarDot(this, thisOffset, that, thatOffset, size)
  }

  companion object {
    private fun vectorDot(
      thiz: F32FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): kotlin.Float {
      var `val` =
        FloatVector.zero(Objects.requireNonNull<VectorSpecies<kotlin.Float>?>(FloatTensor.Companion.F_SPECIES))
      val upperBound: Int = FloatTensor.Companion.F_SPECIES.loopBound(size)
      run {
        var i = 0
        while (i < upperBound) {
          val a = FloatVector.fromMemorySegment(
            FloatTensor.Companion.F_SPECIES,
            thiz.memorySegment,
            (thisOffset + i).toLong() * Float.BYTES,
            ByteOrder.LITTLE_ENDIAN
          )
          val b = FloatVector.fromArray(FloatTensor.Companion.F_SPECIES, that.values, thatOffset + i)
          `val` = a.fma(b, `val`)
          i += FloatTensor.Companion.F_SPECIES.length()
        }
      }
      var result = `val`.reduceLanes(VectorOperators.ADD)
      for (i in upperBound..<size) {
        result += thiz.getFloat((thisOffset + i).toLong()) * that.values[thatOffset + i]
      }
      return result
    }
  }
}
