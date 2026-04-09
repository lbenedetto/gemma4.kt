package com.llama4j.sampler

import com.llama4j.floattensor.FloatTensor

fun interface Sampler {
  fun sampleToken(logits: FloatTensor): Int

  companion object {
    val ARGMAX: Sampler = Sampler { obj: FloatTensor -> obj.argmax() }
  }
}
