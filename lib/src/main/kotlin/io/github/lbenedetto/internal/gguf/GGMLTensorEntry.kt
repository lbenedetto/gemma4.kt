package io.github.lbenedetto.internal.gguf

import java.lang.foreign.MemorySegment

internal data class GGMLTensorEntry(
  val mappedFile: MemorySegment,
  val name: String,
  val ggmlType: GGMLType,
  val shape: IntArray,
  val memorySegment: MemorySegment
)
