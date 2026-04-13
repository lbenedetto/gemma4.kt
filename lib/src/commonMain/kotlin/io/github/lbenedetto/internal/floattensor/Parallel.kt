package io.github.lbenedetto.internal.floattensor

internal expect fun parallelForImpl(startInclusive: Int, endExclusive: Int, action: (Int) -> Unit)

internal fun parallelFor(startInclusive: Int, endExclusive: Int, action: (Int) -> Unit) {
  DebugTimer.measure("parallelFor") {
    parallelForImpl(startInclusive, endExclusive, action)
  }
}
