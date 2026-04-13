package io.github.lbenedetto.internal.floattensor

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlin.experimental.ExperimentalNativeApi

@OptIn(ExperimentalNativeApi::class)
private val NUM_CORES = Platform.getAvailableProcessors().coerceAtLeast(1)

@OptIn(ExperimentalNativeApi::class)
internal actual fun parallelForImpl(
  startInclusive: Int,
  endExclusive: Int,
  action: (Int) -> Unit
) {
  val totalSize = endExclusive - startInclusive
  if (totalSize <= 0) return

  val chunks = minOf(NUM_CORES, totalSize)
  val chunkSize = (totalSize + chunks - 1) / chunks

  runBlocking {
    (0 until chunks).map { chunkIdx ->
      launch(Dispatchers.Default) {
        val chunkStart = startInclusive + (chunkIdx * chunkSize)
        val chunkEnd = minOf(chunkStart + chunkSize, endExclusive)
        for (i in chunkStart until chunkEnd) {
          action(i)
        }
      }
    }.joinAll()
  }
}
