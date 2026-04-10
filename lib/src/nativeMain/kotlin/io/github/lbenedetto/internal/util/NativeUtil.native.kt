package io.github.lbenedetto.internal.util

import kotlinx.cinterop.ExperimentalForeignApi
import platform.posix.fputs
import platform.posix.stderr

@OptIn(ExperimentalForeignApi::class)
internal actual fun printlnStderr(message: String) {
  fputs("$message\n", stderr)
}

/**
 * Never throws exception since we have no mechananism of enabling assertions on native platform
 */
internal actual inline fun assert(condition: Boolean, lazyMessage: () -> Any) {
  if (!condition) {
    printlnStderr(lazyMessage().toString())
  }
}
