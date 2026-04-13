package io.github.lbenedetto.internal.util

import kotlinx.cinterop.ExperimentalForeignApi
import platform.posix.fputs
import platform.posix.stderr

@OptIn(ExperimentalForeignApi::class)
internal actual fun printlnStderr(message: String) {
  fputs("$message\n", stderr)
}
