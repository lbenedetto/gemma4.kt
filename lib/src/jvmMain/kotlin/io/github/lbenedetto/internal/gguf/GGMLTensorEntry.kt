package io.github.lbenedetto.internal.gguf

import java.lang.foreign.MemorySegment
import java.nio.ByteOrder
import java.nio.FloatBuffer

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
      else -> throw UnsupportedOperationException("Conversion to $ggmlType")
    }
  }
}
