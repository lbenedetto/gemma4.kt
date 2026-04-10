package io.github.lbenedetto.internal.gguf

import java.lang.foreign.MemorySegment

internal data class GGMLTensorEntry(
  val ggmlType: GGMLType,
  val shape: IntArray,
  val memorySegment: MemorySegment
)
