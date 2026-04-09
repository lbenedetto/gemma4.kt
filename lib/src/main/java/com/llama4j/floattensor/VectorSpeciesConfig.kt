package com.llama4j.floattensor

import jdk.incubator.vector.VectorShape
import jdk.incubator.vector.VectorSpecies

internal class VectorSpeciesConfig private constructor() {
  val FLOAT: VectorSpecies<Float> =
    VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes<Float>(Float::class.javaPrimitiveType)
  val INT: VectorSpecies<Int> = FLOAT.withLanes<Int>(Int::class.javaPrimitiveType)
  val SHORT_HALF: VectorSpecies<Short> =
    VectorShape.forBitSize(FLOAT.vectorBitSize() / 2).withLanes<Short>(Short::class.javaPrimitiveType)

  companion object {
    val VECTOR_BIT_SIZE: Int = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize())
    val USE_VECTOR_API: Boolean = VECTOR_BIT_SIZE != 0

    fun create(): VectorSpeciesConfig? {
      if (!USE_VECTOR_API) return null
      val config = VectorSpeciesConfig()
      assert(config.FLOAT.length() == config.SHORT_HALF.length())
      return config
    }
  }
}
