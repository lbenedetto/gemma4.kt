package com.llama4j.floattensor

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.ShortVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder

internal class BF16FloatTensor(
  val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor() {

  override fun size(): Long {
    return size
  }

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector? {
    throw UnsupportedOperationException("getFloatVector")
  }

  override fun type(): com.llama4j.gguf.GGMLType {
    return _root_ide_package_.com.llama4j.gguf.GGMLType.BF16
  }

  override fun getFloat(index: Long): Float {
    assert(index in 0..<size)
    val bits: Short = readShort(memorySegment, index * 2)
    return Float.fromBits(bits.toInt() shl 16)
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (USE_VECTOR_API) {
      vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }

  companion object {
    private fun vectorDot(
      thiz: BF16FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      assert(S_SPECIES_HALF!!.length() == F_SPECIES!!.length())
      var `val` =
        FloatVector.zero(F_SPECIES)
      val upperBound: Int = F_SPECIES.loopBound(size)
      var i = 0
      while (i < upperBound) {
        val thatVector = that.getFloatVector(F_SPECIES, thatOffset + i)
        val bfloat16 = ShortVector.fromMemorySegment(
          S_SPECIES_HALF,
          thiz.memorySegment,
          (thisOffset + i) * 2L,
          ByteOrder.LITTLE_ENDIAN
        )
        val thizVector = bfloat16
          .castShape(I_SPECIES!!, 0)
          .lanewise(VectorOperators.LSHL, 16)
          .reinterpretAsFloats()
        `val` = thizVector.fma(thatVector, `val`)
        i += F_SPECIES.length()
      }
      var result = `val`.reduceLanes(VectorOperators.ADD)
      if (upperBound < size) {
        result += scalarDot(
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
