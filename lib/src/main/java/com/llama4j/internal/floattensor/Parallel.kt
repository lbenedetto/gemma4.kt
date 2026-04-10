package com.llama4j.internal.floattensor

import java.util.function.IntConsumer
import java.util.stream.IntStream

internal fun parallelFor(startInclusive: Int, endExclusive: Int, action: IntConsumer) {
  IntStream.range(startInclusive, endExclusive).parallel().forEach(action)
}
