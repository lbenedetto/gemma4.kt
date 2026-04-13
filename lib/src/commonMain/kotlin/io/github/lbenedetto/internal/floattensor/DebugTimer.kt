package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.util.printlnStderr
import kotlin.time.Duration
import kotlin.time.TimeSource

internal object DebugTimer {
  private val timers = mutableMapOf<String, TimerData>()

  @PublishedApi
  internal data class TimerData(
    var totalTime: Duration = Duration.ZERO,
    var count: Long = 0,
  )

  inline fun <T> measure(name: String, block: () -> T): T {
    val mark = TimeSource.Monotonic.markNow()
    val result = block()
    val elapsed = mark.elapsedNow()
    val data = timers.getOrPut(name) { TimerData() }
    data.totalTime += elapsed
    data.count++
    return result
  }

  fun printAndReset() {
    if (timers.isEmpty()) return
    printlnStderr("=== DebugTimer Report ===")
    timers.entries
      .sortedByDescending { it.value.totalTime }
      .forEach { (name, data) ->
        val avg = if (data.count > 0) data.totalTime / data.count.toInt() else Duration.ZERO
        printlnStderr("  $name  total=${ data.totalTime}  count=${data.count}  avg=$avg")
      }
    printlnStderr("=========================")
    timers.clear()
  }
}
