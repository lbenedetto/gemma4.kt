package com.llama4j.util

import java.util.function.IntConsumer
import java.util.stream.IntStream

object Parallel {
  fun parallelFor(startInclusive: Int, endExclusive: Int, action: IntConsumer) {
    IntStream.range(startInclusive, endExclusive).parallel().forEach(action)
  }
}
