package io.github.lbenedetto.internal.model

import io.github.lbenedetto.internal.floattensor.ArrayFloatTensor
import io.github.lbenedetto.internal.floattensor.MutableFloatTensor
import java.util.stream.IntStream

internal class LlamaState internal constructor(config: LlamaConfiguration) {
  val x: MutableFloatTensor // activation at current time stamp (embeddingLength,)
  val xb: MutableFloatTensor // same, but inside a residual branch (embeddingLength,)
  val xbK: MutableFloatTensor // attention output before wo projection (max queryDim,)
  val xb2: MutableFloatTensor // an additional buffer (embeddingLength,)
  val hb: MutableFloatTensor // buffer for hidden dimension in the ffn (maxHiddenDim,)
  val hb2: MutableFloatTensor // buffer for hidden dimension in the ffn (maxHiddenDim,)
  val q: MutableFloatTensor // query (max queryDim,)
  val k: MutableFloatTensor // key (max kvDim,)
  val v: MutableFloatTensor // value (max kvDim,)
  val att: MutableFloatTensor // buffer for scores/attention values (n_heads, seq_len)
  val logits: MutableFloatTensor // output logits

  // kv cache - variable sizes per layer
  val keyCache: Array<MutableFloatTensor> // (n_layer, seq_len, kvDim_per_layer)
  val valueCache: Array<MutableFloatTensor> // (n_layer, seq_len, kvDim_per_layer)

  // per-layer embedding buffers
  val perLayerInputs: MutableFloatTensor?
  val plGate: MutableFloatTensor?
  val plProj: MutableFloatTensor?

  // MoE buffers
  val routerLogits: MutableFloatTensor? // (n_experts,)
  val moeInput: MutableFloatTensor? // (n_embd,) pre-normed MoE input
  val moeOutput: MutableFloatTensor? // (n_embd,) accumulated expert output
  val expertGateUp: MutableFloatTensor? // (2 * expert_ff,)
  val expertDown: MutableFloatTensor? // (n_embd,) single expert output

  var latestToken: Int = 0

  init {
    val maxQueryDim = config.numberOfHeads * config.headSizeFull
    val maxKVDim =
      IntStream.range(0, config.numberOfLayers).map { layer: Int -> config.kvDim(layer) }.max()
        .orElse(0)
    val maxHiddenDim = config.maxHiddenDim()
    this.x = ArrayFloatTensor.allocate(config.embeddingLength)
    this.xb = ArrayFloatTensor.allocate(config.embeddingLength)
    this.xbK = ArrayFloatTensor.allocate(maxQueryDim)
    this.xb2 = ArrayFloatTensor.allocate(config.embeddingLength)
    this.hb = ArrayFloatTensor.allocate(maxHiddenDim)
    this.hb2 = ArrayFloatTensor.allocate(maxHiddenDim)
    this.q = ArrayFloatTensor.allocate(maxQueryDim)
    this.k = ArrayFloatTensor.allocate(maxKVDim)
    this.v = ArrayFloatTensor.allocate(maxKVDim)
    this.att = ArrayFloatTensor.allocate(config.numberOfHeads, config.contextLength)
    this.logits = ArrayFloatTensor.allocate(config.vocabularySize)
    val plDim = config.embeddingLengthPerLayer
    this.perLayerInputs = if (plDim > 0) ArrayFloatTensor.allocate(plDim * config.numberOfLayers) else null
    this.plGate = if (plDim > 0) ArrayFloatTensor.allocate(plDim) else null
    this.plProj = if (plDim > 0) ArrayFloatTensor.allocate(config.embeddingLength) else null
    // MoE buffers
    if (config.isMoE) {
      this.routerLogits = ArrayFloatTensor.allocate(config.expertCount)
      this.moeInput = ArrayFloatTensor.allocate(config.embeddingLength)
      this.moeOutput = ArrayFloatTensor.allocate(config.embeddingLength)
      this.expertGateUp = ArrayFloatTensor.allocate(2 * config.expertFeedForwardLength)
      this.expertDown = ArrayFloatTensor.allocate(config.embeddingLength)
    } else {
      this.routerLogits = null
      this.moeInput = null
      this.moeOutput = null
      this.expertGateUp = null
      this.expertDown = null
    }
    // Only allocate KV caches for layers that have their own KV (not shared)
    val kvDims = IntArray(config.nLayerKvFromStart) { config.kvDim(it) }
    this.keyCache = Array(config.nLayerKvFromStart) { ArrayFloatTensor.allocate(config.contextLength, kvDims[it]) }
    this.valueCache = Array(config.nLayerKvFromStart) { ArrayFloatTensor.allocate(config.contextLength, kvDims[it]) }
  }
}
