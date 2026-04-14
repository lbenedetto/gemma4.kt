package io.github.lbenedetto.internal.sampler

import io.github.lbenedetto.internal.floattensor.FloatTensor
import io.github.lbenedetto.internal.floattensor.MutableFloatTensor

internal fun interface Sampler {
  fun sampleToken(logits: MutableFloatTensor): Int

  companion object {
    val ARGMAX: Sampler = Sampler { obj: FloatTensor -> obj.argmax() }
  }
}
