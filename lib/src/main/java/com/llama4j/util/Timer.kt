package com.llama4j.util

import java.lang.AutoCloseable
import java.util.concurrent.TimeUnit

interface Timer : AutoCloseable {
  override fun close() // no Exception

  companion object {
    @JvmOverloads
    fun log(label: String, timeUnit: TimeUnit = TimeUnit.MILLISECONDS): Timer {
      return object : Timer {
        val startNanos: Long = System.nanoTime()

        override fun close() {
          val elapsedNanos = System.nanoTime() - startNanos
          System.err.println(
            (label + ": "
                + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " "
                + timeUnit.toChronoUnit().name.lowercase())
          )
        }
      }
    }
  }
}
