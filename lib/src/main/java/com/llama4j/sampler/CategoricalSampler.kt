package com.llama4j.sampler

import com.llama4j.floattensor.FloatTensor
import java.util.random.RandomGenerator

@JvmRecord
data class CategoricalSampler(rng: RandomGenerator) : Sampler {
  override fun sampleToken(logits: FloatTensor): Int {
    val random0to1 = rng.nextFloat(1f)
    var cdf = 0.0f
    for (i in 0..<logits.size()) {
      cdf += logits.getFloat(i.toLong())
      if (random0to1 < cdf) {
        return i
      }
    }
    return Math.toIntExact(logits.size()) - 1
  }

  val rng: RandomGenerator

  init {
    this.rng = rng
  }
}
