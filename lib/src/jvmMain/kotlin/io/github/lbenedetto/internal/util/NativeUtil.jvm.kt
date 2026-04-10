package io.github.lbenedetto.internal.util

internal actual fun printlnStderr(message: String) {
  System.err.println(message)
}

internal actual inline fun assert(condition: Boolean, lazyMessage: () -> Any) {
  kotlin.assert(condition, lazyMessage)
}
