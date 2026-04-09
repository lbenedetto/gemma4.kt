package com.llama4j.gguf

import java.lang.foreign.MemorySegment

@JvmRecord
data class GGMLTensorEntry(
  val mappedFile: MemorySegment,
  val name: String,
  val ggmlType: GGMLType,
  val shape: IntArray,
  val memorySegment: MemorySegment
)
