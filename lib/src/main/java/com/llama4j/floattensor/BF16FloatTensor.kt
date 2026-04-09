package com.llama4j.floattensor

import com.llama4j.gguf.GGMLType
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.ShortVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.Float
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder
import java.util.*
import kotlin.Int
import kotlin.Long
import kotlin.Short
import kotlin.UnsupportedOperationException
import kotlin.assert

internal class BF16FloatTensor(size: Long, memorySegment: MemorySegment) : FloatTensor() {
  val size: Long
  val memorySegment: MemorySegment

  init {
    this.size = size
    this.memorySegment = memorySegment
  }

  override fun size(): Long {
    return size
  }

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloatVector(species: VectorSpecies<Float>, index: Int): FloatVector? {
    throw UnsupportedOperationException("getFloatVector")
  }

  public override fun type(): GGMLType {
    return GGMLType.BF16
  }

  override fun getFloat(index: Long): Float {
    assert(0 <= index && index < size)
    val bits: Short = FloatTensor.Companion.readShort(memorySegment, index * 2)
    return Float.intBitsToFloat(bits.toInt() shl 16)
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): kotlin.Float {
    if (FloatTensor.Companion.USE_VECTOR_API) {
      return vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      return FloatTensor.Companion.scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }

  companion object {
    private fun vectorDot(
      thiz: BF16FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): kotlin.Float {
      assert(
        Objects.requireNonNull<VectorSpecies<Short>?>(FloatTensor.Companion.S_SPECIES_HALF)
          .length() == Objects.requireNonNull<VectorSpecies<kotlin.Float>?>(FloatTensor.Companion.F_SPECIES).length()
      )
      var `val` =
        FloatVector.zero(Objects.requireNonNull<VectorSpecies<kotlin.Float>?>(FloatTensor.Companion.F_SPECIES))
      val upperBound: Int = FloatTensor.Companion.F_SPECIES.loopBound(size)
      var i = 0
      while (i < upperBound) {
        val thatVector = that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + i)
        val bfloat16 = ShortVector.fromMemorySegment(
          FloatTensor.Companion.S_SPECIES_HALF,
          thiz.memorySegment,
          (thisOffset + i) * 2L,
          ByteOrder.LITTLE_ENDIAN
        )
        val thizVector = bfloat16
          .castShape<Int>(Objects.requireNonNull<VectorSpecies<Int>?>(FloatTensor.Companion.I_SPECIES), 0)
          .lanewise(VectorOperators.LSHL, 16)
          .reinterpretAsFloats()
        `val` = thizVector.fma(thatVector, `val`)
        i += FloatTensor.Companion.F_SPECIES.length()
      }
      var result = `val`.reduceLanes(VectorOperators.ADD)
      if (upperBound < size) {
        result += FloatTensor.Companion.scalarDot(
          thiz,
          thisOffset + upperBound,
          that,
          thatOffset + upperBound,
          size - upperBound
        )
      }
      return result
    }
  }
}
