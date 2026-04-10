package io.github.lbenedetto.internal.gguf

import io.github.lbenedetto.internal.util.FloatBuffer
import io.github.lbenedetto.internal.util.MemorySegment

internal data class GGMLTensorEntry(
  val ggmlType: GGMLType,
  val shape: IntArray,
  val memorySegment: MemorySegment
) {
  fun toFloatBuffer(): FloatBuffer {
    return when (ggmlType) {
      GGMLType.F32 -> memorySegment.asFloatBuffer()
      else -> throw UnsupportedOperationException("Conversion to $ggmlType")
    }
  }
}
