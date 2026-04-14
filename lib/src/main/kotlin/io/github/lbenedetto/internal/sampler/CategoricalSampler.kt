package io.github.lbenedetto.internal.sampler

import io.github.lbenedetto.internal.floattensor.MutableFloatTensor
import java.util.random.RandomGenerator

@JvmRecord
internal data class CategoricalSampler(val rng: RandomGenerator) : Sampler {
  override fun sampleToken(logits: MutableFloatTensor): Int {
    val random0to1 = rng.nextFloat(1f)
    var cdf = 0.0f
    for (i in 0..<logits.size) {
      cdf += logits.getFloat(i)
      if (random0to1 < cdf) {
        // TODO: Figure out why logits.size is Long but we use the index as an Int to convert to a token
        return i.toInt()
      }
    }
    return Math.toIntExact(logits.size) - 1
  }
}
