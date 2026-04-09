package com.llama4j.model;

import java.util.Arrays;

public final class LlamaConfiguration {
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

  public LlamaConfiguration(int embeddingLength, int[] feedForwardLength, int numberOfLayers,
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

  public boolean isMoE() {
    return expertCount > 0;
  }

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

  public LlamaConfiguration withContextLength(int newContextLength) {
    return new LlamaConfiguration(embeddingLength, feedForwardLength, numberOfLayers,
        numberOfHeads, numberOfKeyValueHeadsPerLayer, vocabularySize,
        newContextLength, rmsNormEps, ropeTheta, ropeThetaSWA,
        headSizeFull, headSizeSWA, slidingWindow,
        logitSoftcapping, isSWA, nLayerKvFromStart,
        embeddingLengthPerLayer,
        expertCount, expertUsedCount, expertFeedForwardLength);
  }
}
