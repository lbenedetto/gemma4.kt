package io.github.lbenedetto.internal.sampler

import io.github.lbenedetto.api.GenerationConfig
import io.github.lbenedetto.internal.floattensor.FloatTensor
import io.github.lbenedetto.internal.util.Math
import kotlin.random.Random

internal fun interface Sampler {
  fun sampleToken(logits: FloatTensor): Int

  companion object {
    val ARGMAX: Sampler = Sampler { it.argmax() }

    internal fun build(vocabularySize: Int, config: GenerationConfig): Sampler {
      if (config.temperature == 0.0f) return Sampler.ARGMAX
      val rng = Random(config.seed)
      val inner: Sampler = if (config.topP <= 0f || config.topP >= 1f) {
        CategoricalSampler(rng)
      } else {
        ToppSampler(vocabularySize, config.topP, rng)
      }
      return Sampler { logits ->
        val size = Math.toIntExact(logits.size)
        logits.divideInPlace(0, size, config.temperature)
        logits.softmaxInPlace(0, size)
        inner.sampleToken(logits)
      }
    }
  }
}
