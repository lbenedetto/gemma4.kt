package io.github.lbenedetto.internal.util

import kotlin.time.DurationUnit
import kotlin.time.TimeSource

internal interface Timer : AutoCloseable {
  override fun close()

  companion object {
    fun log(label: String, timeUnit: DurationUnit = DurationUnit.MILLISECONDS): Timer {
      return object : Timer {
        val mark = TimeSource.Monotonic.markNow()

        override fun close() {
          val elapsed = mark.elapsedNow()
          printlnStderr("$label: ${elapsed.toLong(timeUnit)} ${timeUnit.name.lowercase()}")
        }
      }
    }
  }
}
