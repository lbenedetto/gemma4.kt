package io.github.lbenedetto.cli

actual fun getenv(name: String): String? = System.getenv(name)

actual fun flushStdout() = System.out.flush()

actual fun printStderr(text: String) = System.err.print(text)

actual fun printlnStderr(text: String) = System.err.println(text)

actual fun exitProcess(status: Int): Nothing = kotlin.system.exitProcess(status)
