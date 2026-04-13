package io.github.lbenedetto.internal.sampler

import io.github.lbenedetto.internal.floattensor.FloatTensor

internal fun interface Sampler {
  fun sampleToken(logits: FloatTensor): Int

  companion object {
    val ARGMAX: Sampler = Sampler { obj: FloatTensor -> obj.argmax() }
  }
}
