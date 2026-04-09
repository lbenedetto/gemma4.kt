package com.llama4j.model;

import com.llama4j.floattensor.FloatTensor;
import org.jspecify.annotations.Nullable;

import java.nio.FloatBuffer;

public final class LlamaWeights {
  public final FloatTensor token_embedding_table;
  public final FloatBuffer[] rms_att_weight;       // (layer, dim)
  public final FloatTensor[] wq;                   // (layer, queryDim, dim)
  public final FloatTensor[] wk;                   // (layer, kvDim, dim)
  public final @Nullable FloatTensor[] wv;                   // (layer, kvDim, dim) - null entry if V=K
  public final FloatTensor[] wo;                   // (layer, dim, queryDim)
  public final FloatBuffer[] attn_q_norm;          // (layer, headSize)
  public final FloatBuffer[] attn_k_norm;          // (layer, headSize)
  public final FloatBuffer[] post_attention_norm;  // (layer, dim)
  public final FloatBuffer[] rms_ffn_weight;       // (layer, dim) - shared MLP norm
  public final FloatTensor[] w1;                   // ffn_gate (layer, hiddenDim, dim)
  public final FloatTensor[] w2;                   // ffn_down (layer, dim, hiddenDim)
  public final FloatTensor[] w3;                   // ffn_up (layer, hiddenDim, dim)
  public final FloatBuffer[] post_ffw_norm;        // (layer, dim) - overall post-FFW norm
  public final FloatBuffer rms_final_weight;
  public final float[] layerOutputScale;
  // Full attention RoPE
  public final FloatBuffer freq_cis_real_full;
  public final FloatBuffer freq_cis_imag_full;
  // SWA RoPE
  public final FloatBuffer freq_cis_real_swa;
  public final FloatBuffer freq_cis_imag_swa;
  public final FloatTensor wcls;
  // Per-layer embedding weights
  public final @Nullable FloatTensor perLayerTokenEmbd;
  public final @Nullable FloatTensor perLayerModelProj;
  public final @Nullable FloatBuffer perLayerProjNorm;
  public final FloatTensor @Nullable [] perLayerInpGate;
  public final FloatTensor @Nullable [] perLayerProj;
  public final FloatBuffer @Nullable [] perLayerPostNorm;
  // MoE weights (null if dense model)
  public final @Nullable FloatTensor @Nullable [] ffnGateInp;        // router weight (layer, n_experts, n_embd)
  public final FloatBuffer @Nullable [] ffnGateInpScale;   // router input scale (layer, n_embd)
  public final FloatTensor @Nullable [] ffnGateUpExps;     // fused gate+up expert (layer, n_experts * 2*expert_ff, n_embd)
  public final FloatTensor @Nullable [] ffnDownExps;       // down expert (layer, n_experts * n_embd, expert_ff)
  public final FloatBuffer @Nullable [] ffnDownExpsScale;  // expert output scale (layer, n_experts)
  public final FloatBuffer @Nullable [] ffnPostNorm1;      // shared MLP post norm (layer, dim) - MoE only
  public final FloatBuffer @Nullable [] preFfwNorm2;       // MoE pre-norm (layer, dim)
  public final FloatBuffer @Nullable [] ffnPostNorm2;      // MoE post norm (layer, dim)

  public LlamaWeights(
          FloatTensor token_embedding_table,
          FloatBuffer[] rms_att_weight,
          FloatTensor[] wq,
          FloatTensor[] wk,
          @Nullable FloatTensor[] wv,
          FloatTensor[] wo,
          FloatBuffer[] attn_q_norm,
          FloatBuffer[] attn_k_norm,
          FloatBuffer[] post_attention_norm,
          FloatBuffer[] rms_ffn_weight,
          FloatTensor[] w1,
          FloatTensor[] w2,
          FloatTensor[] w3,
          FloatBuffer[] post_ffw_norm,
          FloatBuffer rms_final_weight,
          float[] layerOutputScale,
          FloatBuffer freq_cis_real_full,
          FloatBuffer freq_cis_imag_full,
          FloatBuffer freq_cis_real_swa,
          FloatBuffer freq_cis_imag_swa,
          FloatTensor wcls,
          @Nullable FloatTensor perLayerTokenEmbd,
          @Nullable FloatTensor perLayerModelProj,
          @Nullable FloatBuffer perLayerProjNorm,
          FloatTensor @Nullable [] perLayerInpGate,
          FloatTensor @Nullable [] perLayerProj,
          FloatBuffer @Nullable [] perLayerPostNorm,
          FloatTensor @Nullable [] ffnGateInp,
          FloatBuffer @Nullable [] ffnGateInpScale,
          FloatTensor @Nullable [] ffnGateUpExps,
          FloatTensor @Nullable [] ffnDownExps,
          FloatBuffer @Nullable [] ffnDownExpsScale,
          FloatBuffer @Nullable [] ffnPostNorm1,
          FloatBuffer @Nullable [] preFfwNorm2,
          FloatBuffer @Nullable [] ffnPostNorm2
  ) {
    this.token_embedding_table = token_embedding_table;
    this.rms_att_weight = rms_att_weight;
    this.wq = wq;
    this.wk = wk;
    this.wv = wv;
    this.wo = wo;
    this.attn_q_norm = attn_q_norm;
    this.attn_k_norm = attn_k_norm;
    this.post_attention_norm = post_attention_norm;
    this.rms_ffn_weight = rms_ffn_weight;
    this.w1 = w1;
    this.w2 = w2;
    this.w3 = w3;
    this.post_ffw_norm = post_ffw_norm;
    this.rms_final_weight = rms_final_weight;
    this.layerOutputScale = layerOutputScale;
    this.freq_cis_real_full = freq_cis_real_full;
    this.freq_cis_imag_full = freq_cis_imag_full;
    this.freq_cis_real_swa = freq_cis_real_swa;
    this.freq_cis_imag_swa = freq_cis_imag_swa;
    this.wcls = wcls;
    this.perLayerTokenEmbd = perLayerTokenEmbd;
    this.perLayerModelProj = perLayerModelProj;
    this.perLayerProjNorm = perLayerProjNorm;
    this.perLayerInpGate = perLayerInpGate;
    this.perLayerProj = perLayerProj;
    this.perLayerPostNorm = perLayerPostNorm;
    this.ffnGateInp = ffnGateInp;
    this.ffnGateInpScale = ffnGateInpScale;
    this.ffnGateUpExps = ffnGateUpExps;
    this.ffnDownExps = ffnDownExps;
    this.ffnDownExpsScale = ffnDownExpsScale;
    this.ffnPostNorm1 = ffnPostNorm1;
    this.preFfwNorm2 = preFfwNorm2;
    this.ffnPostNorm2 = ffnPostNorm2;
  }
}
