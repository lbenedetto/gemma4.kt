package io.github.lbenedetto.internal.floattensor


import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.F_SPECIES
import io.github.lbenedetto.internal.util.vectorMathEnabled
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies

actual object ArrayFloatTensorMath {

  internal actual fun vectorDot(
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

internal fun ArrayFloatTensor.getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector {
  if (!vectorMathEnabled()) {
    throw UnsupportedOperationException()
  }
  return FloatVector.fromArray(F_SPECIES, values, offset)
}
