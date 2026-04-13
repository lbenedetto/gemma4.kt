package io.github.lbenedetto.internal.sampler

import io.github.lbenedetto.internal.floattensor.FloatTensor
import io.github.lbenedetto.internal.util.Math
import kotlin.random.Random

internal data class CategoricalSampler(val rng: Random) : Sampler {
  override fun sampleToken(logits: FloatTensor): Int {
    val random0to1 = rng.nextFloat()
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
