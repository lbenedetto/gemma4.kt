package io.github.lbenedetto.internal.floattensor

import java.util.stream.IntStream

internal actual fun parallelFor(
  startInclusive: Int,
  endExclusive: Int,
  action: (Int) -> Unit
) {
  IntStream.range(startInclusive, endExclusive).parallel().forEach(action)
}
