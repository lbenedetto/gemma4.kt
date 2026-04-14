package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.F_SPECIES
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.I_SPECIES
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.S_SPECIES_HALF
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.USE_VECTOR_API
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.readShort
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.scalarDot
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.ShortVector
import jdk.incubator.vector.VectorOperators
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder

internal class BF16FloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor {

  override fun get(index: Long): Float {
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
      var value =
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
