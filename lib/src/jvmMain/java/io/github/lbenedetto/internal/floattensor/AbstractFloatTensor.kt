package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.gguf.GGMLType
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies

internal abstract class AbstractFloatTensor : FloatTensor {
  abstract fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector?

  abstract fun type(): GGMLType?
}
