package com.llama4j.sampler

import com.llama4j.floattensor.FloatTensor
import java.util.function.ToDoubleFunction
import java.util.random.RandomGenerator

class ToppSampler(maxNumberOfElements: Int, topp: Float, rng: RandomGenerator) : Sampler {
  val indices: IntArray
  val topp: Float
  val rng: RandomGenerator

  init {
    this.indices = IntArray(maxNumberOfElements)
    this.topp = topp
    this.rng = rng
  }

  override fun sampleToken(logits: FloatTensor): Int {
    val comparator =
      Comparator.comparingDouble<Int>(ToDoubleFunction { i: Int -> logits.getFloat(i.toLong()) }).reversed()

    val n = Math.toIntExact(logits.size())
    var head = 0
    var tail = n - 1
    val cutoff = (1.0f - topp) / (n - 1)
    for (i in indices.indices) {
      if (logits.getFloat(i.toLong()) >= cutoff) {
        indices[head++] = i
      } else {
        indices[tail--] = i
      }
    }

    val n0 = head
    for (i in n0 / 2 - 1 downTo 0) {
      siftDown(indices, i, n0, comparator)
    }

    var cumulativeProb = 0.0f
    var lastIndex = 0
    for (i in n0 - 1 downTo 0) {
      swap(indices, 0, i)
      cumulativeProb += logits.getFloat(indices[i].toLong())
      if (cumulativeProb > topp) {
        lastIndex = i
        break
      }
      siftDown(indices, 0, i - 1, comparator)
    }

    val r = rng.nextFloat(1f) * cumulativeProb
    var cdf = 0.0f
    for (i in n0 - 1 downTo lastIndex) {
      cdf += logits.getFloat(indices[i].toLong())
      if (r < cdf) {
        return indices[i]
      }
    }

    return indices[lastIndex]
  }

  companion object {
    fun swap(array: IntArray, from: Int, to: Int) {
      val tmp = array[from]
      array[from] = array[to]
      array[to] = tmp
    }

    fun siftDown(array: IntArray, from: Int, n: Int, comparator: Comparator<Int>) {
      var prev = from
      var next: Int
      while (((2 * prev + 1).also { next = it }) < n) {
        val r = 2 * prev + 2
        if (r < n && comparator.compare(array[r], array[next]) < 0) {
          next = r
        }
        if (comparator.compare(array[next], array[prev]) < 0) {
          swap(array, prev, next)
          prev = next
        } else {
          break
        }
      }
    }
  }
}
