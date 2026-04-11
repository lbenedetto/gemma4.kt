package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.F_SPECIES
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.I_SPECIES
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.S_SPECIES_HALF
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.USE_VECTOR_API
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.util.MemorySegment
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.ShortVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.nio.ByteOrder

internal class F16FloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : AbstractFloatTensor() {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector? {
    throw UnsupportedOperationException("getFloatVector")
  }

  override fun type(): GGMLType {
    return GGMLType.F16
  }

  override fun getFloat(index: Long): Float {
    assert(index in 0..<size)
    return memorySegment.readFloat16(index * 2)
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
      thiz: F16FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      assert(
        S_SPECIES_HALF!!
          .length() == F_SPECIES!!.length()
      )
      var value = FloatVector.zero(F_SPECIES)
      val upperBound: Int = F_SPECIES.loopBound(size)
      var i = 0
      while (i < upperBound) {
        val thatVector = that.getFloatVector(F_SPECIES, thatOffset + i)
        val bits16 = ShortVector.fromMemorySegment(
          S_SPECIES_HALF,
          thiz.memorySegment.actual(),
          (thisOffset + i) * 2L,
          ByteOrder.LITTLE_ENDIAN
        )
        var bits32 =
          bits16.castShape(I_SPECIES!!, 0)
            .reinterpretAsInts()
        val zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31)
        bits32 = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16)
          .or(bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13).and(zeroExponentMask))
        val thizVector = bits32.reinterpretAsFloats()
        value = thizVector.fma(thatVector, value)
        i += F_SPECIES.length()
      }
      var result = value.reduceLanes(VectorOperators.ADD)
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
