package com.llama4j.gguf

import com.llama4j.model.Llama

@JvmRecord
internal data class PartialModel(
  val modelFileName: String,
  val model: Llama,
  val tensorDataOffset: Long,
  val tensorInfos: MutableMap<String, GGUFTensorInfo>,
  val ropeFreqsSWA: Pair<FloatArray, FloatArray>,
  val ropeFreqsFull: Pair<FloatArray, FloatArray>
)
