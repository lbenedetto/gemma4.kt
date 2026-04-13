package io.github.lbenedetto.internal.floattensor

import jdk.incubator.vector.VectorShape
import jdk.incubator.vector.VectorSpecies

val VECTOR_BIT_SIZE: Int = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize())
val USE_VECTOR_API: Boolean = VECTOR_BIT_SIZE != 0

object VectorSpeciesConfig {
  val VECTOR_BIT_SIZE: Int = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize())
  val USE_VECTOR_API: Boolean = VECTOR_BIT_SIZE != 0

  val F_SPECIES: VectorSpecies<Float>?
  val I_SPECIES: VectorSpecies<Int>?
  val S_SPECIES_HALF: VectorSpecies<Short>?

  init {
    if (USE_VECTOR_API) {
      F_SPECIES =
        VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(Float::class.javaPrimitiveType)
      I_SPECIES = F_SPECIES.withLanes(Int::class.javaPrimitiveType)
      S_SPECIES_HALF =
        VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(Short::class.javaPrimitiveType)
      assert(F_SPECIES.length() == S_SPECIES_HALF.length())
    } else {
      F_SPECIES = null
      I_SPECIES = null
      S_SPECIES_HALF = null
    }
  }
}
