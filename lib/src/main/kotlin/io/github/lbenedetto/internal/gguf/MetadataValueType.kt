package io.github.lbenedetto.internal.gguf

internal enum class MetadataValueType {
  UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64;

  companion object {
    private val VALUES = entries.toTypedArray()

    fun fromIndex(index: Int): MetadataValueType {
      return VALUES[index]
    }
  }
}
