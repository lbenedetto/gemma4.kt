package com.llama4j.model;

import com.llama4j.floattensor.ArrayFloatTensor;
import com.llama4j.floattensor.FloatTensor;

import java.util.stream.IntStream;

public final class LlamaState {
  public final FloatTensor x;      // activation at current time stamp (embeddingLength,)
  public final FloatTensor xb;     // same, but inside a residual branch (embeddingLength,)
  public final FloatTensor xb_k;   // attention output before wo projection (max queryDim,)
  public final FloatTensor xb2;    // an additional buffer (embeddingLength,)
  public final FloatTensor hb;     // buffer for hidden dimension in the ffn (maxHiddenDim,)
  public final FloatTensor hb2;    // buffer for hidden dimension in the ffn (maxHiddenDim,)
  public final FloatTensor q;      // query (max queryDim,)
  public final FloatTensor k;      // key (max kvDim,)
  public final FloatTensor v;      // value (max kvDim,)
  public final FloatTensor att;    // buffer for scores/attention values (n_heads, seq_len)
  public final FloatTensor logits; // output logits
  // kv cache - variable sizes per layer
  public final FloatTensor[] keyCache;   // (n_layer, seq_len, kvDim_per_layer)
  public final FloatTensor[] valueCache; // (n_layer, seq_len, kvDim_per_layer)
  // per-layer embedding buffers
  public final FloatTensor perLayerInputs;
  public final FloatTensor plGate;
  public final FloatTensor plProj;
  // MoE buffers
  public final FloatTensor routerLogits;    // (n_experts,)
  public final FloatTensor moeInput;        // (n_embd,) pre-normed MoE input
  public final FloatTensor moeOutput;       // (n_embd,) accumulated expert output
  public final FloatTensor expertGateUp;    // (2 * expert_ff,)
  public final FloatTensor expertDown;      // (n_embd,) single expert output

  public int latestToken;

  LlamaState(LlamaConfiguration config) {
    int maxQueryDim = config.numberOfHeads * config.headSizeFull;
    int maxKVDim = IntStream.range(0, config.numberOfLayers).map(config::kvDim).max().orElse(0);
    int maxHiddenDim = config.maxHiddenDim();
    this.x = ArrayFloatTensor.allocate(config.embeddingLength);
    this.xb = ArrayFloatTensor.allocate(config.embeddingLength);
    this.xb_k = ArrayFloatTensor.allocate(maxQueryDim);
    this.xb2 = ArrayFloatTensor.allocate(config.embeddingLength);
    this.hb = ArrayFloatTensor.allocate(maxHiddenDim);
    this.hb2 = ArrayFloatTensor.allocate(maxHiddenDim);
    this.q = ArrayFloatTensor.allocate(maxQueryDim);
    this.k = ArrayFloatTensor.allocate(maxKVDim);
    this.v = ArrayFloatTensor.allocate(maxKVDim);
    this.att = ArrayFloatTensor.allocate(config.numberOfHeads, config.contextLength);
    this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
    int plDim = config.embeddingLengthPerLayer;
    this.perLayerInputs = plDim > 0 ? ArrayFloatTensor.allocate(plDim * config.numberOfLayers) : null;
    this.plGate = plDim > 0 ? ArrayFloatTensor.allocate(plDim) : null;
    this.plProj = plDim > 0 ? ArrayFloatTensor.allocate(config.embeddingLength) : null;
    // MoE buffers
    if (config.isMoE()) {
      this.routerLogits = ArrayFloatTensor.allocate(config.expertCount);
      this.moeInput = ArrayFloatTensor.allocate(config.embeddingLength);
      this.moeOutput = ArrayFloatTensor.allocate(config.embeddingLength);
      this.expertGateUp = ArrayFloatTensor.allocate(2 * config.expertFeedForwardLength);
      this.expertDown = ArrayFloatTensor.allocate(config.embeddingLength);
    } else {
      this.routerLogits = null;
      this.moeInput = null;
      this.moeOutput = null;
      this.expertGateUp = null;
      this.expertDown = null;
    }
    // Only allocate KV caches for layers that have their own KV (not shared)
    this.keyCache = new FloatTensor[config.nLayerKvFromStart];
    this.valueCache = new FloatTensor[config.nLayerKvFromStart];
    for (int l = 0; l < config.nLayerKvFromStart; l++) {
      int kvDim = config.kvDim(l);
      keyCache[l] = ArrayFloatTensor.allocate(config.contextLength, kvDim);
      valueCache[l] = ArrayFloatTensor.allocate(config.contextLength, kvDim);
    }
  }
}
