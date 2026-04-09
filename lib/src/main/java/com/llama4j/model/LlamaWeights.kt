package com.llama4j.model

import com.llama4j.floattensor.FloatTensor
import java.nio.FloatBuffer

class LlamaWeights(
  val token_embedding_table: FloatTensor,
// (layer, dim)
  val rms_att_weight: Array<FloatBuffer>,
  wq: Array<FloatTensor>,
  wk: Array<FloatTensor>,
  wv: Array<FloatTensor?>,
  wo: Array<FloatTensor>,
  attn_q_norm: Array<FloatBuffer>,
  attn_k_norm: Array<FloatBuffer>,
  post_attention_norm: Array<FloatBuffer>,
  rms_ffn_weight: Array<FloatBuffer>,
  w1: Array<FloatTensor>,
  w2: Array<FloatTensor>,
  w3: Array<FloatTensor>,
  post_ffw_norm: Array<FloatBuffer>,
  rms_final_weight: FloatBuffer,
  layerOutputScale: FloatArray,
  freq_cis_real_full: FloatBuffer,
  freq_cis_imag_full: FloatBuffer,
  freq_cis_real_swa: FloatBuffer,
  freq_cis_imag_swa: FloatBuffer,
  wcls: FloatTensor,
  perLayerTokenEmbd: FloatTensor?,
  perLayerModelProj: FloatTensor?,
  perLayerProjNorm: FloatBuffer?,
  perLayerInpGate: Array<FloatTensor>?,
  perLayerProj: Array<FloatTensor>?,
  perLayerPostNorm: Array<FloatBuffer>?,
  ffnGateInp: Array<FloatTensor>?,
  ffnGateInpScale: Array<FloatBuffer>?,
  ffnGateUpExps: Array<FloatTensor>?,
  ffnDownExps: Array<FloatTensor>?,
  ffnDownExpsScale: Array<FloatBuffer>?,
  ffnPostNorm1: Array<FloatBuffer>?,
  preFfwNorm2: Array<FloatBuffer>?,
  ffnPostNorm2: Array<FloatBuffer>?
) {
  val wq: Array<FloatTensor> // (layer, queryDim, dim)
  val wk: Array<FloatTensor> // (layer, kvDim, dim)
  val wv: Array<FloatTensor?> // (layer, kvDim, dim) - null entry if V=K
  val wo: Array<FloatTensor> // (layer, dim, queryDim)
  val attn_q_norm: Array<FloatBuffer> // (layer, headSize)
  val attn_k_norm: Array<FloatBuffer> // (layer, headSize)
  val post_attention_norm: Array<FloatBuffer> // (layer, dim)
  val rms_ffn_weight: Array<FloatBuffer> // (layer, dim) - shared MLP norm
  val w1: Array<FloatTensor> // ffn_gate (layer, hiddenDim, dim)
  val w2: Array<FloatTensor> // ffn_down (layer, dim, hiddenDim)
  val w3: Array<FloatTensor> // ffn_up (layer, hiddenDim, dim)
  val post_ffw_norm: Array<FloatBuffer> // (layer, dim) - overall post-FFW norm
  val rms_final_weight: FloatBuffer
  val layerOutputScale: FloatArray

  // Full attention RoPE
  val freq_cis_real_full: FloatBuffer
  val freq_cis_imag_full: FloatBuffer

  // SWA RoPE
  val freq_cis_real_swa: FloatBuffer
  val freq_cis_imag_swa: FloatBuffer
  val wcls: FloatTensor

  // Per-layer embedding weights
  val perLayerTokenEmbd: FloatTensor?
  val perLayerModelProj: FloatTensor?
  val perLayerProjNorm: FloatBuffer?
  val perLayerInpGate: Array<FloatTensor>?
  val perLayerProj: Array<FloatTensor>?
  val perLayerPostNorm: Array<FloatBuffer>?

  // MoE weights (null if dense model)
  val ffnGateInp: Array<FloatTensor>? // router weight (layer, n_experts, n_embd)
  val ffnGateInpScale: Array<FloatBuffer>? // router input scale (layer, n_embd)
  val ffnGateUpExps: Array<FloatTensor>? // fused gate+up expert (layer, n_experts * 2*expert_ff, n_embd)
  val ffnDownExps: Array<FloatTensor>? // down expert (layer, n_experts * n_embd, expert_ff)
  val ffnDownExpsScale: Array<FloatBuffer>? // expert output scale (layer, n_experts)
  val ffnPostNorm1: Array<FloatBuffer>? // shared MLP post norm (layer, dim) - MoE only
  val preFfwNorm2: Array<FloatBuffer>? // MoE pre-norm (layer, dim)
  val ffnPostNorm2: Array<FloatBuffer>? // MoE post norm (layer, dim)

  init {
    this.wq = wq
    this.wk = wk
    this.wv = wv
    this.wo = wo
    this.attn_q_norm = attn_q_norm
    this.attn_k_norm = attn_k_norm
    this.post_attention_norm = post_attention_norm
    this.rms_ffn_weight = rms_ffn_weight
    this.w1 = w1
    this.w2 = w2
    this.w3 = w3
    this.post_ffw_norm = post_ffw_norm
    this.rms_final_weight = rms_final_weight
    this.layerOutputScale = layerOutputScale
    this.freq_cis_real_full = freq_cis_real_full
    this.freq_cis_imag_full = freq_cis_imag_full
    this.freq_cis_real_swa = freq_cis_real_swa
    this.freq_cis_imag_swa = freq_cis_imag_swa
    this.wcls = wcls
    this.perLayerTokenEmbd = perLayerTokenEmbd
    this.perLayerModelProj = perLayerModelProj
    this.perLayerProjNorm = perLayerProjNorm
    this.perLayerInpGate = perLayerInpGate
    this.perLayerProj = perLayerProj
    this.perLayerPostNorm = perLayerPostNorm
    this.ffnGateInp = ffnGateInp
    this.ffnGateInpScale = ffnGateInpScale
    this.ffnGateUpExps = ffnGateUpExps
    this.ffnDownExps = ffnDownExps
    this.ffnDownExpsScale = ffnDownExpsScale
    this.ffnPostNorm1 = ffnPostNorm1
    this.preFfwNorm2 = preFfwNorm2
    this.ffnPostNorm2 = ffnPostNorm2
  }
}
