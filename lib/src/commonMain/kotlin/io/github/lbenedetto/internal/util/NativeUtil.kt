package io.github.lbenedetto.internal.util

internal expect fun printlnStderr(message: String)
internal expect fun printStderr(message: String)

internal fun assert(value: Boolean) {
  assert(value) { "Assertion failed" }
}

/**
 * Throws an [AssertionError] calculated by [lazyMessage] if the [condition] is false
 * and runtime assertions have been enabled on the JVM using the *-ea* JVM option.
 */
internal expect inline fun assert(condition: Boolean, lazyMessage: () -> Any)

expect fun vectorMathEnabled(): Boolean
