package io.github.lbenedetto.internal.model

import io.github.lbenedetto.internal.floattensor.FloatTensor
import java.nio.FloatBuffer

class LlamaWeights(
  val tokenEmbeddingTable: FloatTensor,
  // (layer, dim)
  val rmsAttWeight: Array<FloatBuffer>,
  // (layer, queryDim, dim)
  val wq: Array<FloatTensor>,
  // (layer, kvDim, dim)
  val wk: Array<FloatTensor>,
  // (layer, kvDim, dim) - null entry if V=K
  val wv: Array<FloatTensor?>,
  // (layer, dim, queryDim)
  val wo: Array<FloatTensor>,
  // (layer, headSize)
  val attnQNorm: Array<FloatBuffer>,
  // (layer, headSize)
  val attnKNorm: Array<FloatBuffer>,
  // (layer, dim)
  val postAttentionNorm: Array<FloatBuffer>,
  // (layer, dim) - shared MLP norm
  val rmsFfnWeight: Array<FloatBuffer>,
  // ffn_gate (layer, hiddenDim, dim)
  val w1: Array<FloatTensor>,
  // ffn_down (layer, dim, hiddenDim)
  val w2: Array<FloatTensor>,
  // ffn_up (layer, hiddenDim, dim)
  val w3: Array<FloatTensor>,
  // (layer, dim) - overall post-FFW norm
  val postFfwNorm: Array<FloatBuffer>,
  val rmsFinalWeight: FloatBuffer,
  val layerOutputScale: FloatArray,
  // Full attention RoPE
  val freqCisRealFull: FloatBuffer,
  val freqCisImagFull: FloatBuffer,
  // SWA RoPE
  val freqCisRealSwa: FloatBuffer,
  val freqCisImagSwa: FloatBuffer,
  val wcls: FloatTensor,
  // Per-layer embedding weights
  val perLayerTokenEmbd: FloatTensor?,
  val perLayerModelProj: FloatTensor?,
  val perLayerProjNorm: FloatBuffer?,
  val perLayerInpGate: Array<FloatTensor>?,
  val perLayerProj: Array<FloatTensor>?,
  val perLayerPostNorm: Array<FloatBuffer>?,
  // MoE weights (null if dense model)
  val ffnGateInp: Array<FloatTensor>?, // router weight (layer, n_experts, n_embd)
  // router input scale (layer, n_embd)
  val ffnGateInpScale: Array<FloatBuffer>?,
  // fused gate+up expert (layer, n_experts * 2*expert_ff, n_embd)
  val ffnGateUpExps: Array<FloatTensor>?,
  // down expert (layer, n_experts * n_embd, expert_ff)
  val ffnDownExps: Array<FloatTensor>?,
  // expert output scale (layer, n_experts)
  val ffnDownExpsScale: Array<FloatBuffer>?,
  // shared MLP post norm (layer, dim) - MoE only
  val ffnPostNorm1: Array<FloatBuffer>?,
  // MoE pre-norm (layer, dim)
  val preFfwNorm2: Array<FloatBuffer>?,
  // MoE post norm (layer, dim)
  val ffnPostNorm2: Array<FloatBuffer>?
)
