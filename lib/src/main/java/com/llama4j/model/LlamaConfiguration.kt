package com.llama4j.model

import java.util.*

class LlamaConfiguration(
  embeddingLength: Int, feedForwardLength: IntArray, numberOfLayers: Int,
  numberOfHeads: Int, numberOfKeyValueHeadsPerLayer: IntArray, vocabularySize: Int,
  contextLength: Int, rmsNormEps: Float, ropeTheta: Float, ropeThetaSWA: Float,
  headSizeFull: Int, headSizeSWA: Int, slidingWindow: Int,
  logitSoftcapping: Float, isSWA: BooleanArray, nLayerKvFromStart: Int,
  embeddingLengthPerLayer: Int,
  expertCount: Int, expertUsedCount: Int, expertFeedForwardLength: Int
) {
  val embeddingLength: Int
  val feedForwardLength: IntArray // per-layer (shared MLP)
  val numberOfLayers: Int
  val numberOfHeads: Int
  val numberOfKeyValueHeadsPerLayer: IntArray // per-layer KV head count
  val vocabularySize: Int
  val contextLength: Int
  val rmsNormEps: Float
  val ropeTheta: Float // full attention RoPE theta
  val ropeThetaSWA: Float // SWA RoPE theta
  val headSizeFull: Int // head size for full attention layers
  val headSizeSWA: Int // head size for SWA layers
  val slidingWindow: Int
  val logitSoftcapping: Float
  val isSWA: BooleanArray // per-layer: true = SWA, false = full attention
  val nLayerKvFromStart: Int // first N layers have own KV cache, rest reuse
  val embeddingLengthPerLayer: Int // per-layer embedding dim (0 = disabled)

  // MoE fields
  val expertCount: Int // 0 = dense model (no MoE)
  val expertUsedCount: Int // top-k experts per token
  val expertFeedForwardLength: Int // expert FFN hidden dim

  init {
    this.embeddingLength = embeddingLength
    this.feedForwardLength = feedForwardLength
    this.numberOfLayers = numberOfLayers
    this.numberOfHeads = numberOfHeads
    this.numberOfKeyValueHeadsPerLayer = numberOfKeyValueHeadsPerLayer
    this.vocabularySize = vocabularySize
    this.contextLength = contextLength
    this.rmsNormEps = rmsNormEps
    this.ropeTheta = ropeTheta
    this.ropeThetaSWA = ropeThetaSWA
    this.headSizeFull = headSizeFull
    this.headSizeSWA = headSizeSWA
    this.slidingWindow = slidingWindow
    this.logitSoftcapping = logitSoftcapping
    this.isSWA = isSWA
    this.nLayerKvFromStart = nLayerKvFromStart
    this.embeddingLengthPerLayer = embeddingLengthPerLayer
    this.expertCount = expertCount
    this.expertUsedCount = expertUsedCount
    this.expertFeedForwardLength = expertFeedForwardLength
  }

  val isMoE: Boolean
    get() = expertCount > 0

  // For layers without own KV, return the layer whose cache to reuse
  fun kvSourceLayer(layer: Int): Int {
    if (layer < nLayerKvFromStart) return layer // has own KV

    // Reuse the last KV layer of the same attention type
    return nLayerKvFromStart - (if (isSWA[layer]) 2 else 1)
  }

  fun hasKv(layer: Int): Boolean {
    return layer < nLayerKvFromStart
  }

  fun headSize(layer: Int): Int {
    return if (isSWA[layer]) headSizeSWA else headSizeFull
  }

  fun numberOfKeyValueHeads(layer: Int): Int {
    return numberOfKeyValueHeadsPerLayer[layer]
  }

  fun kvDim(layer: Int): Int {
    return numberOfKeyValueHeadsPerLayer[layer] * headSize(layer)
  }

  fun queryDim(layer: Int): Int {
    return numberOfHeads * headSize(layer)
  }

  fun maxHiddenDim(): Int {
    return Arrays.stream(feedForwardLength).max().orElseThrow()
  }

  fun withContextLength(newContextLength: Int): LlamaConfiguration {
    return LlamaConfiguration(
      embeddingLength, feedForwardLength, numberOfLayers,
      numberOfHeads, numberOfKeyValueHeadsPerLayer, vocabularySize,
      newContextLength, rmsNormEps, ropeTheta, ropeThetaSWA,
      headSizeFull, headSizeSWA, slidingWindow,
      logitSoftcapping, isSWA, nLayerKvFromStart,
      embeddingLengthPerLayer,
      expertCount, expertUsedCount, expertFeedForwardLength
    )
  }
}
