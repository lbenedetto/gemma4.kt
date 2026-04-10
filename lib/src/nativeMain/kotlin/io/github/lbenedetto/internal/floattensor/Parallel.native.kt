package io.github.lbenedetto.internal.floattensor

internal actual fun parallelFor(
  startInclusive: Int,
  endExclusive: Int,
  action: (Int) -> Unit
) {
  // TODO: Native parallelFor
  for (i in startInclusive until endExclusive) {
    action(i)
  }
}
