package com.llama4j.internal.gguf

import com.llama4j.internal.model.Llama

@JvmRecord
internal data class PartialModel(
  val modelFileName: String,
  val model: Llama,
  val tensorDataOffset: Long,
  val tensorInfos: MutableMap<String, GGUFTensorInfo>,
  val ropeFreqsSWA: Pair<FloatArray, FloatArray>,
  val ropeFreqsFull: Pair<FloatArray, FloatArray>
)
