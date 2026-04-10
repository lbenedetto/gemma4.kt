package io.github.lbenedetto.internal.floattensor

internal expect fun parallelFor(startInclusive: Int, endExclusive: Int, action: (Int) -> Unit)
