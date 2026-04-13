package io.github.lbenedetto.internal.gguf

@JvmRecord
internal data class GGUFTensorInfo(
  val name: String,
  val dimensions: IntArray,
  val ggmlType: GGMLType,
  val offset: Long
)
