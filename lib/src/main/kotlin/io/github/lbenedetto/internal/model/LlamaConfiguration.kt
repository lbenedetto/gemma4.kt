package io.github.lbenedetto.internal.model

import java.util.*

internal class LlamaConfiguration(
  val embeddingLength: Int, // per-layer (shared MLP)
  val feedForwardLength: IntArray, val numberOfLayers: Int,
  val numberOfHeads: Int, // per-layer KV head count
  val numberOfKeyValueHeadsPerLayer: IntArray, val vocabularySize: Int,
  val contextLength: Int, val rmsNormEps: Float, // full attention RoPE theta
  val ropeTheta: Float, // SWA RoPE theta
  val ropeThetaSWA: Float,
  // head size for full attention layers
  val headSizeFull: Int, // head size for SWA layers
  val headSizeSWA: Int, val slidingWindow: Int,
  val logitSoftcapping: Float, // per-layer: true = SWA, false = full attention
  val isSWA: BooleanArray, // first N layers have own KV cache, rest reuse
  val nLayerKvFromStart: Int,
  // per-layer embedding dim (0 = disabled)
  val embeddingLengthPerLayer: Int,
  // MoE fields
  val expertCount: Int, // 0 = dense model (no MoE)
  // top-k experts per token
  val expertUsedCount: Int, // expert FFN hidden dim
  val expertFeedForwardLength: Int
) {

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
