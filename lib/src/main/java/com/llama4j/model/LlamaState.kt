package com.llama4j.model

import com.llama4j.floattensor.ArrayFloatTensor
import com.llama4j.floattensor.FloatTensor
import java.util.function.IntUnaryOperator
import java.util.stream.IntStream

class LlamaState internal constructor(config: LlamaConfiguration) {
  val x: FloatTensor // activation at current time stamp (embeddingLength,)
  val xb: FloatTensor // same, but inside a residual branch (embeddingLength,)
  val xb_k: FloatTensor // attention output before wo projection (max queryDim,)
  val xb2: FloatTensor // an additional buffer (embeddingLength,)
  val hb: FloatTensor // buffer for hidden dimension in the ffn (maxHiddenDim,)
  val hb2: FloatTensor // buffer for hidden dimension in the ffn (maxHiddenDim,)
  val q: FloatTensor // query (max queryDim,)
  val k: FloatTensor // key (max kvDim,)
  val v: FloatTensor // value (max kvDim,)
  val att: FloatTensor // buffer for scores/attention values (n_heads, seq_len)
  val logits: FloatTensor // output logits

  // kv cache - variable sizes per layer
  val keyCache: Array<FloatTensor> // (n_layer, seq_len, kvDim_per_layer)
  val valueCache: Array<FloatTensor> // (n_layer, seq_len, kvDim_per_layer)

  // per-layer embedding buffers
  val perLayerInputs: FloatTensor?
  val plGate: FloatTensor?
  val plProj: FloatTensor?

  // MoE buffers
  val routerLogits: FloatTensor? // (n_experts,)
  val moeInput: FloatTensor? // (n_embd,) pre-normed MoE input
  val moeOutput: FloatTensor? // (n_embd,) accumulated expert output
  val expertGateUp: FloatTensor? // (2 * expert_ff,)
  val expertDown: FloatTensor? // (n_embd,) single expert output

  var latestToken: Int = 0

  init {
    val maxQueryDim = config.numberOfHeads * config.headSizeFull
    val maxKVDim =
      IntStream.range(0, config.numberOfLayers).map(IntUnaryOperator { layer: Int -> config.kvDim(layer) }).max()
        .orElse(0)
    val maxHiddenDim = config.maxHiddenDim()
    this.x = ArrayFloatTensor.Companion.allocate(config.embeddingLength)
    this.xb = ArrayFloatTensor.Companion.allocate(config.embeddingLength)
    this.xb_k = ArrayFloatTensor.Companion.allocate(maxQueryDim)
    this.xb2 = ArrayFloatTensor.Companion.allocate(config.embeddingLength)
    this.hb = ArrayFloatTensor.Companion.allocate(maxHiddenDim)
    this.hb2 = ArrayFloatTensor.Companion.allocate(maxHiddenDim)
    this.q = ArrayFloatTensor.Companion.allocate(maxQueryDim)
    this.k = ArrayFloatTensor.Companion.allocate(maxKVDim)
    this.v = ArrayFloatTensor.Companion.allocate(maxKVDim)
    this.att = ArrayFloatTensor.Companion.allocate(config.numberOfHeads, config.contextLength)
    this.logits = ArrayFloatTensor.Companion.allocate(config.vocabularySize)
    val plDim = config.embeddingLengthPerLayer
    this.perLayerInputs = if (plDim > 0) ArrayFloatTensor.Companion.allocate(plDim * config.numberOfLayers) else null
    this.plGate = if (plDim > 0) ArrayFloatTensor.Companion.allocate(plDim) else null
    this.plProj = if (plDim > 0) ArrayFloatTensor.Companion.allocate(config.embeddingLength) else null
    // MoE buffers
    if (config.isMoE()) {
      this.routerLogits = ArrayFloatTensor.Companion.allocate(config.expertCount)
      this.moeInput = ArrayFloatTensor.Companion.allocate(config.embeddingLength)
      this.moeOutput = ArrayFloatTensor.Companion.allocate(config.embeddingLength)
      this.expertGateUp = ArrayFloatTensor.Companion.allocate(2 * config.expertFeedForwardLength)
      this.expertDown = ArrayFloatTensor.Companion.allocate(config.embeddingLength)
    } else {
      this.routerLogits = null
      this.moeInput = null
      this.moeOutput = null
      this.expertGateUp = null
      this.expertDown = null
    }
    // Only allocate KV caches for layers that have their own KV (not shared)
    this.keyCache = arrayOfNulls<FloatTensor>(config.nLayerKvFromStart)
    this.valueCache = arrayOfNulls<FloatTensor>(config.nLayerKvFromStart)
    for (l in 0..<config.nLayerKvFromStart) {
      val kvDim = config.kvDim(l)
      keyCache[l] = ArrayFloatTensor.Companion.allocate(config.contextLength, kvDim)
      valueCache[l] = ArrayFloatTensor.Companion.allocate(config.contextLength, kvDim)
    }
  }
}
