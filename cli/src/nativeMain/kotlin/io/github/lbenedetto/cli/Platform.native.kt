package io.github.lbenedetto.cli

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import platform.posix.fflush
import platform.posix.fprintf
import platform.posix.stderr
import platform.posix.stdout

@OptIn(ExperimentalForeignApi::class)
actual fun getenv(name: String): String? = platform.posix.getenv(name)?.toKString()

@OptIn(ExperimentalForeignApi::class)
actual fun flushStdout() { fflush(stdout) }

@OptIn(ExperimentalForeignApi::class)
actual fun printStderr(text: String) { fprintf(stderr, "%s", text) }

@OptIn(ExperimentalForeignApi::class)
actual fun printlnStderr(text: String) { fprintf(stderr, "%s\n", text) }

actual fun exitProcess(status: Int): Nothing = kotlin.system.exitProcess(status)
