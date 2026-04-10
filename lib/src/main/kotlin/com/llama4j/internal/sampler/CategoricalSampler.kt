package com.llama4j.internal.sampler

import com.llama4j.internal.floattensor.FloatTensor
import java.util.random.RandomGenerator

@JvmRecord
data class CategoricalSampler(val rng: RandomGenerator) : Sampler {
  override fun sampleToken(logits: FloatTensor): Int {
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
