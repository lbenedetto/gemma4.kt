package io.github.lbenedetto.internal.gguf

import io.github.lbenedetto.internal.util.FloatBuffer
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder

internal data class GGMLTensorEntry(
  val ggmlType: GGMLType,
  val shape: IntArray,
  val memorySegment: MemorySegment
) {
  fun toFloatBuffer(): FloatBuffer {
    return when (ggmlType) {
      GGMLType.F32 -> memorySegment.asByteBuffer()
        .order(ByteOrder.LITTLE_ENDIAN)
        .asFloatBuffer()
        .let { FloatBuffer(it) }
      else -> throw UnsupportedOperationException("Conversion to $ggmlType")
    }
  }
}
