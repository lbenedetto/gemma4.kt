package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.F_SPECIES
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import java.nio.ByteOrder

object F32FloatTensorMath {
  internal fun vectorDot(
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
          thiz.memorySegment.actual(),
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
