package io.github.lbenedetto.cli

import kotlin.time.TimeSource

expect fun getenv(name: String): String?
expect fun flushStdout()
expect fun printStderr(text: String)
expect fun printlnStderr(text: String = "")
expect fun exitProcess(status: Int): Nothing

fun defaultSeed(): Long = TimeSource.Monotonic.markNow().hashCode().toLong()
