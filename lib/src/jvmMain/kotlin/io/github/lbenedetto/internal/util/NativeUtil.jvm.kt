package io.github.lbenedetto.internal.util

import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig

internal actual fun printlnStderr(message: String) {
  System.err.println(message)
}

internal actual inline fun assert(condition: Boolean, lazyMessage: () -> Any) {
  kotlin.assert(condition, lazyMessage)
}

actual fun vectorMathEnabled(): Boolean = VectorSpeciesConfig.USE_VECTOR_API
