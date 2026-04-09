package com.llama4j;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

record Llama(Configuration configuration, GemmaTokenizer tokenizer, Weights weights) {
    public State createNewState() {
        State state = new State(configuration());
        state.latestToken = tokenizer.getSpecialTokens().get("<bos>");
        return state;
    }

    public static final class Configuration {
        public final int embeddingLength;
        public final int[] feedForwardLength; // per-layer (shared MLP)
        public final int numberOfLayers;
        public final int numberOfHeads;
        public final int[] numberOfKeyValueHeadsPerLayer; // per-layer KV head count
        public final int vocabularySize;
        public final int contextLength;
        public final float rmsNormEps;
        public final float ropeTheta;       // full attention RoPE theta
        public final float ropeThetaSWA;    // SWA RoPE theta
        public final int headSizeFull;      // head size for full attention layers
        public final int headSizeSWA;       // head size for SWA layers
        public final int slidingWindow;
        public final float logitSoftcapping;
        public final boolean[] isSWA;       // per-layer: true = SWA, false = full attention
        public final int nLayerKvFromStart; // first N layers have own KV cache, rest reuse
        public final int embeddingLengthPerLayer; // per-layer embedding dim (0 = disabled)
        // MoE fields
        public final int expertCount;           // 0 = dense model (no MoE)
        public final int expertUsedCount;       // top-k experts per token
        public final int expertFeedForwardLength; // expert FFN hidden dim

        public Configuration(int embeddingLength, int[] feedForwardLength, int numberOfLayers,
                             int numberOfHeads, int[] numberOfKeyValueHeadsPerLayer, int vocabularySize,
                             int contextLength, float rmsNormEps, float ropeTheta, float ropeThetaSWA,
                             int headSizeFull, int headSizeSWA, int slidingWindow,
                             float logitSoftcapping, boolean[] isSWA, int nLayerKvFromStart,
                             int embeddingLengthPerLayer,
                             int expertCount, int expertUsedCount, int expertFeedForwardLength) {
            this.embeddingLength = embeddingLength;
            this.feedForwardLength = feedForwardLength;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeadsPerLayer = numberOfKeyValueHeadsPerLayer;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.ropeThetaSWA = ropeThetaSWA;
            this.headSizeFull = headSizeFull;
            this.headSizeSWA = headSizeSWA;
            this.slidingWindow = slidingWindow;
            this.logitSoftcapping = logitSoftcapping;
            this.isSWA = isSWA;
            this.nLayerKvFromStart = nLayerKvFromStart;
            this.embeddingLengthPerLayer = embeddingLengthPerLayer;
            this.expertCount = expertCount;
            this.expertUsedCount = expertUsedCount;
            this.expertFeedForwardLength = expertFeedForwardLength;
        }

        public boolean isMoE() { return expertCount > 0; }

        // For layers without own KV, return the layer whose cache to reuse
        public int kvSourceLayer(int layer) {
            if (layer < nLayerKvFromStart) return layer; // has own KV
            // Reuse the last KV layer of the same attention type
            return nLayerKvFromStart - (isSWA[layer] ? 2 : 1);
        }

        public boolean hasKv(int layer) {
            return layer < nLayerKvFromStart;
        }

        public int headSize(int layer) {
            return isSWA[layer] ? headSizeSWA : headSizeFull;
        }

        public int numberOfKeyValueHeads(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer];
        }

        public int kvDim(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer] * headSize(layer);
        }

        public int queryDim(int layer) {
            return numberOfHeads * headSize(layer);
        }

        public int maxHiddenDim() {
            return Arrays.stream(feedForwardLength).max().orElseThrow();
        }

        public Configuration withContextLength(int newContextLength) {
            return new Configuration(embeddingLength, feedForwardLength, numberOfLayers,
                    numberOfHeads, numberOfKeyValueHeadsPerLayer, vocabularySize,
                    newContextLength, rmsNormEps, ropeTheta, ropeThetaSWA,
                    headSizeFull, headSizeSWA, slidingWindow,
                    logitSoftcapping, isSWA, nLayerKvFromStart,
                    embeddingLengthPerLayer,
                    expertCount, expertUsedCount, expertFeedForwardLength);
        }
    }

    public static final class Weights {
        public final FloatTensor token_embedding_table;
        public final FloatBuffer[] rms_att_weight;       // (layer, dim)
        public final FloatTensor[] wq;                   // (layer, queryDim, dim)
        public final FloatTensor[] wk;                   // (layer, kvDim, dim)
        public final FloatTensor[] wv;                   // (layer, kvDim, dim) - null entry if V=K
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
        public final FloatTensor perLayerTokenEmbd;
        public final FloatTensor perLayerModelProj;
        public final FloatBuffer perLayerProjNorm;
        public final FloatTensor[] perLayerInpGate;
        public final FloatTensor[] perLayerProj;
        public final FloatBuffer[] perLayerPostNorm;
        // MoE weights (null if dense model)
        public final FloatTensor[] ffnGateInp;        // router weight (layer, n_experts, n_embd)
        public final FloatBuffer[] ffnGateInpScale;   // router input scale (layer, n_embd)
        public final FloatTensor[] ffnGateUpExps;     // fused gate+up expert (layer, n_experts * 2*expert_ff, n_embd)
        public final FloatTensor[] ffnDownExps;       // down expert (layer, n_experts * n_embd, expert_ff)
        public final FloatBuffer[] ffnDownExpsScale;  // expert output scale (layer, n_experts)
        public final FloatBuffer[] ffnPostNorm1;      // shared MLP post norm (layer, dim) - MoE only
        public final FloatBuffer[] preFfwNorm2;       // MoE pre-norm (layer, dim)
        public final FloatBuffer[] ffnPostNorm2;      // MoE post norm (layer, dim)

        public Weights(FloatTensor token_embedding_table,
                       FloatBuffer[] rms_att_weight,
                       FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo,
                       FloatBuffer[] attn_q_norm, FloatBuffer[] attn_k_norm,
                       FloatBuffer[] post_attention_norm,
                       FloatBuffer[] rms_ffn_weight,
                       FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3,
                       FloatBuffer[] post_ffw_norm,
                       FloatBuffer rms_final_weight,
                       float[] layerOutputScale,
                       FloatBuffer freq_cis_real_full, FloatBuffer freq_cis_imag_full,
                       FloatBuffer freq_cis_real_swa, FloatBuffer freq_cis_imag_swa,
                       FloatTensor wcls,
                       FloatTensor perLayerTokenEmbd, FloatTensor perLayerModelProj,
                       FloatBuffer perLayerProjNorm,
                       FloatTensor[] perLayerInpGate, FloatTensor[] perLayerProj,
                       FloatBuffer[] perLayerPostNorm,
                       FloatTensor[] ffnGateInp, FloatBuffer[] ffnGateInpScale,
                       FloatTensor[] ffnGateUpExps, FloatTensor[] ffnDownExps,
                       FloatBuffer[] ffnDownExpsScale,
                       FloatBuffer[] ffnPostNorm1, FloatBuffer[] preFfwNorm2,
                       FloatBuffer[] ffnPostNorm2) {
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

    public static final class State {
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

        State(Configuration config) {
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

    static float gelu(float x) {
        return (float) (0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))));
    }

    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        final float finalss = ss;
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }

    static void rmsnorm(FloatTensor out, int outOffset, FloatTensor x, int xOffset, FloatBuffer weight, int size, float rmsNormEps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(xOffset + i);
            ss += xi * xi;
        }
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        for (int i = 0; i < size; i++) {
            out.setFloat(outOffset + i, weight.get(i) * ss * x.getFloat(xOffset + i));
        }
    }

    // Bare RMS norm without learned weights (just normalize to unit RMS)
    static void rmsnormNoWeight(FloatTensor out, int outOffset, FloatTensor x, int xOffset, int size, float rmsNormEps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(xOffset + i);
            ss += xi * xi;
        }
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        for (int i = 0; i < size; i++) {
            out.setFloat(outOffset + i, ss * x.getFloat(xOffset + i));
        }
    }

    static FloatTensor forward(Llama model, Llama.State state, int token, int position) {
        Llama.Configuration config = model.configuration();
        Llama.Weights weights = model.weights();
        int dim = config.embeddingLength;
        float sqrtDim = (float) Math.sqrt(dim);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        state.x.mapInPlace(v -> v * sqrtDim);

        // Compute per-layer inputs (if model has per-layer embeddings)
        int plDim = config.embeddingLengthPerLayer;
        int plTotal = plDim * config.numberOfLayers;
        if (plDim > 0 && weights.perLayerTokenEmbd != null) {
            float sqrtPlDim = (float) Math.sqrt(plDim);
            float projScale = (float) (1.0 / Math.sqrt(dim));
            float inputScale = (float) (1.0 / Math.sqrt(2.0));

            // Project x through perLayerModelProj, scale, and RMS norm per chunk
            weights.perLayerModelProj.matmul(state.x, state.perLayerInputs, plTotal, dim);
            state.perLayerInputs.mapInPlace(0, plTotal, v -> v * projScale);
            for (int l = 0; l < config.numberOfLayers; l++) {
                rmsnorm(state.perLayerInputs, l * plDim, state.perLayerInputs, l * plDim,
                        weights.perLayerProjNorm, plDim, config.rmsNormEps);
            }

            // Add per-layer token embedding scaled by sqrt(plDim)
            long tokEmbOffset = (long) token * plTotal;
            for (int i = 0; i < plTotal; i++) {
                float tokEmb = weights.perLayerTokenEmbd.getFloat(tokEmbOffset + i) * sqrtPlDim;
                state.perLayerInputs.setFloat(i, state.perLayerInputs.getFloat(i) + tokEmb);
            }

            // Scale combined input by 1/sqrt(2)
            state.perLayerInputs.mapInPlace(0, plTotal, v -> v * inputScale);
        }

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            boolean layerIsSWA = config.isSWA[l];
            int headSize = config.headSize(l);
            int halfHead = headSize / 2;
            int queryDim = config.queryDim(l);
            int kvDim = config.kvDim(l);
            int hiddenDim = config.feedForwardLength[l];

            // attention rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], dim, config.rmsNormEps);

            // Q projection + per-head RMS norm
            weights.wq[l].matmul(state.xb, state.q, queryDim, dim);
            for (int h = 0; h < config.numberOfHeads; h++) {
                rmsnorm(state.q, h * headSize, state.q, h * headSize, weights.attn_q_norm[l], headSize, config.rmsNormEps);
            }

            // RoPE (NeoX style: SWA layers use different frequencies than full attention)
            FloatBuffer freqsReal = layerIsSWA ? weights.freq_cis_real_swa : weights.freq_cis_real_full;
            FloatBuffer freqsImag = layerIsSWA ? weights.freq_cis_imag_swa : weights.freq_cis_imag_full;
            for (int h = 0; h < config.numberOfHeads; ++h) {
                int poffset = h * headSize;
                for (int i0 = 0; i0 < headSize; i0 += 2) {
                    int ic = i0 / 2;
                    float fcr = freqsReal.get(position * halfHead + ic);
                    float fci = freqsImag.get(position * halfHead + ic);
                    float v0 = state.q.getFloat(poffset + ic);
                    float v1 = state.q.getFloat(poffset + ic + halfHead);
                    state.q.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                    state.q.setFloat(poffset + ic + halfHead, v0 * fci + v1 * fcr);
                }
            }

            // KV projection (shared KV: later layers reuse earlier layer's cache)
            int kvLayer = config.kvSourceLayer(l);
            int nKvHeads = config.numberOfKeyValueHeads(l);
            int kvMul = config.numberOfHeads / nKvHeads;
            if (config.hasKv(l)) {
                weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
                // V = wv @ xb if V weight exists, otherwise V = K
                if (weights.wv[l] != null) {
                    weights.wv[l].matmul(state.xb, state.v, kvDim, dim);
                } else {
                    state.k.copyTo(0, state.v, 0, kvDim);
                }

                // Per-head K norm (learned weights) and V norm (bare RMS)
                for (int h = 0; h < nKvHeads; h++) {
                    rmsnorm(state.k, h * headSize, state.k, h * headSize, weights.attn_k_norm[l], headSize, config.rmsNormEps);
                    rmsnormNoWeight(state.v, h * headSize, state.v, h * headSize, headSize, config.rmsNormEps);
                }

                // RoPE for K
                for (int h = 0; h < nKvHeads; ++h) {
                    int poffset = h * headSize;
                    for (int i0 = 0; i0 < headSize; i0 += 2) {
                        int ic = i0 / 2;
                        float fcr = freqsReal.get(position * halfHead + ic);
                        float fci = freqsImag.get(position * halfHead + ic);
                        float v0 = state.k.getFloat(poffset + ic);
                        float v1 = state.k.getFloat(poffset + ic + halfHead);
                        state.k.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                        state.k.setFloat(poffset + ic + halfHead, v0 * fci + v1 * fcr);
                    }
                }

                state.k.copyTo(0, state.keyCache[kvLayer], position * kvDim, kvDim);
                state.v.copyTo(0, state.valueCache[kvLayer], position * kvDim, kvDim);
            }

            // Attention (scale=1.0, no 1/sqrt(headSize))
            int attStart = layerIsSWA ? Math.max(0, position - config.slidingWindow + 1) : 0;
            int finalKvLayer = kvLayer;
            int finalKvDim = kvDim;
            int finalAttStart = attStart;

            Parallel.parallelFor(0, config.numberOfHeads, h -> {
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength;
                for (int t = finalAttStart; t <= position; t++) {
                    int keyCacheOffset = t * finalKvDim + (h / kvMul) * headSize;
                    float score = state.q.dot(qOffset, state.keyCache[finalKvLayer], keyCacheOffset, headSize);
                    state.att.setFloat(attOffset + t, score);
                }

                state.att.softmaxInPlace(attOffset + finalAttStart, position - finalAttStart + 1);
                int xbOffset = h * headSize;
                state.xb_k.fillInPlace(xbOffset, headSize, 0f);
                for (int t = finalAttStart; t <= position; t++) {
                    int vOffset = t * finalKvDim + (h / kvMul) * headSize;
                    float a = state.att.getFloat(attOffset + t);
                    state.xb_k.saxpyInPlace(xbOffset, state.valueCache[finalKvLayer], vOffset, headSize, a);
                }
            });

            // Output projection + post-attention norm + residual
            weights.wo[l].matmul(state.xb_k, state.xb2, dim, queryDim);
            rmsnorm(state.xb2, state.xb2, weights.post_attention_norm[l], dim, config.rmsNormEps);
            state.x.addInPlace(state.xb2);

            // FFN
            boolean isMoELayer = config.isMoE() && weights.ffnGateInp[l] != null;
            if (isMoELayer) {
                // === MoE FFN: shared MLP + expert MoE ===

                // Shared MLP path: ffn_norm -> gate/up/down -> post_norm_1
                rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], dim, config.rmsNormEps);
                weights.w1[l].matmul(state.xb, state.hb, hiddenDim, dim);
                weights.w3[l].matmul(state.xb, state.hb2, hiddenDim, dim);
                state.hb.mapInPlace(0, hiddenDim, Llama::gelu);
                state.hb.multiplyInPlace(0, state.hb2, 0, hiddenDim);
                weights.w2[l].matmul(state.hb, state.xb, dim, hiddenDim);
                rmsnorm(state.xb, state.xb, weights.ffnPostNorm1[l], dim, config.rmsNormEps);
                // state.xb now holds shared MLP output

                // Expert MoE path: pre_norm_2 -> routing -> expert FFN -> post_norm_2
                rmsnorm(state.moeInput, state.x, weights.preFfwNorm2[l], dim, config.rmsNormEps);

                // Router: rms_norm(x) -> scale(1/sqrt(dim)) -> elementwise_mul(gate_inp_s) -> matmul(gate_inp) -> softmax -> top-k
                {
                    float ss = state.x.reduce(0, dim, 0f, (acc, xi) -> acc + xi * xi);
                    ss /= dim;
                    ss += config.rmsNormEps;
                    float rmsScale = (float) (1.0 / Math.sqrt(ss)) / (float) Math.sqrt(dim);
                    for (int i = 0; i < dim; i++) {
                        state.xb2.setFloat(i, state.x.getFloat(i) * rmsScale * weights.ffnGateInpScale[l].get(i));
                    }
                }
                weights.ffnGateInp[l].matmul(state.xb2, state.routerLogits, config.expertCount, dim);

                // Softmax over router logits
                state.routerLogits.softmaxInPlace(0, config.expertCount);

                // Find top-k experts
                int nExperts = config.expertCount;
                int topK = config.expertUsedCount;
                int[] topExperts = new int[topK];
                float[] topProbs = new float[topK];
                for (int ki = 0; ki < topK; ki++) {
                    int bestIdx = 0;
                    float bestVal = Float.NEGATIVE_INFINITY;
                    for (int ei = 0; ei < nExperts; ei++) {
                        float val = state.routerLogits.getFloat(ei);
                        if (val > bestVal) {
                            bestVal = val;
                            bestIdx = ei;
                        }
                    }
                    topExperts[ki] = bestIdx;
                    topProbs[ki] = bestVal;
                    state.routerLogits.setFloat(bestIdx, Float.NEGATIVE_INFINITY); // mask for next iteration
                }

                // Run selected experts and accumulate
                int expertFF = config.expertFeedForwardLength;
                int gateUpDim = 2 * expertFF;
                state.moeOutput.fillInPlace(0, dim, 0f);

                for (int ki = 0; ki < topK; ki++) {
                    int expertIdx = topExperts[ki];
                    float prob = topProbs[ki];
                    float downScale = weights.ffnDownExpsScale[l].get(expertIdx);

                    // gate_up = ffn_gate_up_exps[expert] @ moeInput -> (2*expertFF,)
                    int gateUpOffset = expertIdx * gateUpDim * dim;
                    weights.ffnGateUpExps[l].matmul(state.moeInput, state.expertGateUp, gateUpDim, dim, gateUpOffset);

                    // gate = gelu(gate_up[0:expertFF]), up = gate_up[expertFF:2*expertFF]
                    state.expertGateUp.mapInPlace(0, expertFF, Llama::gelu);
                    for (int i = 0; i < expertFF; i++) {
                        state.expertGateUp.setFloat(i, state.expertGateUp.getFloat(i) * state.expertGateUp.getFloat(expertFF + i));
                    }

                    // down = ffn_down_exps[expert] @ (gate * up) -> (dim,)
                    int downOffset = expertIdx * dim * expertFF;
                    weights.ffnDownExps[l].matmul(state.expertGateUp, state.expertDown, dim, expertFF, downOffset);

                    // Accumulate: moeOutput += prob * downScale * expertDown
                    float finalWeight = prob * downScale;
                    state.moeOutput.saxpyInPlace(0, state.expertDown, 0, dim, finalWeight);
                }

                // Post-norm for MoE output
                rmsnorm(state.moeOutput, state.moeOutput, weights.ffnPostNorm2[l], dim, config.rmsNormEps);

                // Combine shared MLP + MoE: xb += moeOutput
                state.xb.addInPlace(0, state.moeOutput, 0, dim);

                // Overall post-FFW norm + residual
                rmsnorm(state.xb, state.xb, weights.post_ffw_norm[l], dim, config.rmsNormEps);
                state.x.addInPlace(state.xb);
            } else {
                // Standard dense FFN: w2(GELU(w1(x)) * w3(x))
                rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], dim, config.rmsNormEps);
                weights.w1[l].matmul(state.xb, state.hb, hiddenDim, dim);
                weights.w3[l].matmul(state.xb, state.hb2, hiddenDim, dim);
                state.hb.mapInPlace(0, hiddenDim, Llama::gelu);
                state.hb.multiplyInPlace(0, state.hb2, 0, hiddenDim);
                weights.w2[l].matmul(state.hb, state.xb, dim, hiddenDim);
                rmsnorm(state.xb, state.xb, weights.post_ffw_norm[l], dim, config.rmsNormEps);
                state.x.addInPlace(state.xb);
            }

            // Per-layer embedding: GELU-gated projection
            if (plDim > 0 && weights.perLayerInpGate != null) {
                weights.perLayerInpGate[l].matmul(state.x, state.plGate, plDim, dim);
                state.plGate.mapInPlace(0, plDim, Llama::gelu);
                int plOffset = l * plDim;
                for (int i = 0; i < plDim; i++) {
                    state.plGate.setFloat(i, state.plGate.getFloat(i) * state.perLayerInputs.getFloat(plOffset + i));
                }
                weights.perLayerProj[l].matmul(state.plGate, state.plProj, dim, plDim);
                rmsnorm(state.plProj, state.plProj, weights.perLayerPostNorm[l], dim, config.rmsNormEps);
                state.x.addInPlace(state.plProj);
            }

            // Layer output scale
            float scale = weights.layerOutputScale[l];
            if (scale != 1.0f) {
                state.x.mapInPlace(0, dim, v -> v * scale);
            }
        }

        // Final norm + logits
        rmsnorm(state.x, state.x, weights.rms_final_weight, dim, config.rmsNormEps);
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);
        if (config.logitSoftcapping > 0) {
            float cap = config.logitSoftcapping;
            state.logits.mapInPlace(v -> cap * (float) Math.tanh(v / cap));
        }
        return state.logits;
    }

    private static final String ANSI_CYAN = "\033[36m";
    private static final String ANSI_RESET = "\033[0m";

    public static List<Integer> generateTokens(Llama model, Llama.State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
                                               boolean color, IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        long startGen = 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // current token (initialized to BOS)
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            forward(model, state, token, position);
            if (promptIndex < promptTokens.size()) {
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    System.err.print(GemmaTokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
                if (promptIndex >= promptTokens.size()) {
                    startGen = System.nanoTime();
                }
            } else {
                nextToken = sampler.sampleToken(state.logits);
                if (echo) {
                    System.err.print(GemmaTokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
                generatedTokens.add(nextToken);
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        String timingPrefix = color ? ANSI_CYAN : "";
        String timingSuffix = color ? ANSI_RESET : "";
        System.err.printf("%n%scontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%s%n",
                timingPrefix,
                startPosition + promptIndex + generatedTokens.size(), model.configuration().contextLength,
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(),
                generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size(),
                timingSuffix);

        return generatedTokens;
    }
}
