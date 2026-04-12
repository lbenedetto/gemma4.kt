package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.F_SPECIES
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.I_SPECIES
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.S_SPECIES_HALF
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.ShortVector
import jdk.incubator.vector.VectorOperators
import java.nio.ByteOrder

actual object BF16FloatTensorMath {
  internal actual fun vectorDot(
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
      val thatVector = that.getFloatVector(thatOffset + i)
      val bfloat16 = ShortVector.fromMemorySegment(
        S_SPECIES_HALF,
        thiz.memorySegment.actual(),
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
