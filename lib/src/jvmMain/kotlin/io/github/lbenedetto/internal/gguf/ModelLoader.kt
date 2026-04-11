package io.github.lbenedetto.internal.gguf

import io.github.lbenedetto.internal.floattensor.FloatTensor
import io.github.lbenedetto.internal.floattensor.FloatTensorFactory
import io.github.lbenedetto.internal.model.Llama
import io.github.lbenedetto.internal.model.LlamaConfiguration
import io.github.lbenedetto.internal.model.LlamaWeights
import io.github.lbenedetto.internal.model.RoPE
import io.github.lbenedetto.internal.tokenizer.GemmaTokenizer
import io.github.lbenedetto.internal.tokenizer.Vocabulary
import io.github.lbenedetto.internal.util.FloatBuffer
import io.github.lbenedetto.internal.util.Timer
import io.github.lbenedetto.internal.util.wrapWithFloatBuffer
import okio.Path
import java.util.*

internal object ModelLoader {
  private fun loadVocabulary(metadata: Map<String, Any>): Vocabulary {
    val tokens = metadata["tokenizer.ggml.tokens"]!! as Array<String>
    val scores = metadata["tokenizer.ggml.scores"] as FloatArray
    return Vocabulary(tokens, scores)
  }

  fun loadModel(ggufPath: Path, contextLength: Int): Llama {
    Timer.log("Load Gemma4 model").use {
      val gguf: GGUF = GGUF.loadModel(ggufPath)
      return loadModel(ggufPath, gguf, contextLength, true)
    }
  }

  fun loadModel(ggufPath: Path, gguf: GGUF, contextLength: Int, loadWeightsFlag: Boolean): Llama {
    var contextLength = contextLength
    val metadata = gguf.metadata

    val vocabulary = loadVocabulary(metadata)
    val tokenizer = createTokenizer(metadata, vocabulary)

    val modelContextLength = metadata["gemma4.context_length"] as Int
    if (contextLength !in 0..modelContextLength) {
      contextLength = modelContextLength
    }

    val embeddingLength = metadata["gemma4.embedding_length"] as Int
    val numberOfHeads = metadata["gemma4.attention.head_count"] as Int
    val numberOfLayers = metadata["gemma4.block_count"] as Int

    val headSizeFull = metadata["gemma4.attention.key_length"] as Int
    val headSizeSWA = metadata["gemma4.attention.key_length_swa"] as Int
    val slidingWindow = metadata["gemma4.attention.sliding_window"] as Int
    val logitSoftcapping = metadata.getOrDefault("gemma4.final_logit_softcapping", 0f) as Float
    val rmsNormEps = metadata.getOrDefault("gemma4.attention.layer_norm_rms_epsilon", 1e-6f) as Float
    val ropeTheta = metadata.getOrDefault("gemma4.rope.freq_base", 1000000f) as Float
    val ropeThetaSWA = metadata.getOrDefault("gemma4.rope.freq_base_swa", 10000f) as Float

    // MoE parameters
    val expertCount = metadata.getOrDefault("gemma4.expert_count", 0) as Int
    val expertUsedCount = metadata.getOrDefault("gemma4.expert_used_count", 0) as Int
    val expertFeedForwardLength = metadata.getOrDefault("gemma4.expert_feed_forward_length", 0) as Int

    // Per-layer feed forward lengths (scalar for uniform, array for variable)
    val feedForwardLength: IntArray
    val ffnRaw = metadata["gemma4.feed_forward_length"]
    if (ffnRaw is IntArray) {
      feedForwardLength = ffnRaw
    } else {
      feedForwardLength = IntArray(numberOfLayers)
      Arrays.fill(feedForwardLength, ffnRaw!! as Int)
    }

    val tensorInfos = gguf.tensorInfos

    // Derive isSWA per layer from Q norm weight size (256 = SWA, 512 = full attention)
    val isSWA: BooleanArray
    val swaRaw = metadata["gemma4.attention.sliding_window_pattern"]
    if (swaRaw is BooleanArray) {
      isSWA = swaRaw
    } else {
      // Derive from tensor shapes: check Q norm size per layer
      isSWA = BooleanArray(numberOfLayers)
      for (i in 0..<numberOfLayers) {
        val qNorm = tensorInfos["blk.$i.attn_q_norm.weight"]
        if (qNorm != null) {
          val qNormSize: Long = FloatTensor.numberOfElementsLong(*qNorm.dimensions)
          isSWA[i] = (qNormSize == headSizeSWA.toLong())
        } else {
          isSWA[i] = (i % 5 != 4) // fallback
        }
      }
    }

    // Derive per-layer KV head count from K weight shapes
    val numberOfKeyValueHeadsPerLayer = IntArray(numberOfLayers)
    for (i in 0..<numberOfLayers) {
      val kWeight = tensorInfos["blk.$i.attn_k.weight"]
      val headSize = if (isSWA[i]) headSizeSWA else headSizeFull
      if (kWeight != null) {
        val kDim = kWeight.dimensions[1].toLong()
        numberOfKeyValueHeadsPerLayer[i] = (kDim / headSize).toInt()
      } else {
        numberOfKeyValueHeadsPerLayer[i] = numberOfHeads // fallback
      }
    }

    // Shared KV layers: last N layers reuse KV from earlier layers
    val sharedKvLayers = metadata.getOrDefault("gemma4.attention.shared_kv_layers", 0) as Int
    val nLayerKvFromStart = numberOfLayers - sharedKvLayers

    val embeddingLengthPerLayer = metadata.getOrDefault("gemma4.embedding_length_per_layer_input", 0) as Int

    val config = LlamaConfiguration(
      embeddingLength,
      feedForwardLength,
      numberOfLayers,
      numberOfHeads,
      numberOfKeyValueHeadsPerLayer,
      vocabulary.size(),
      contextLength,
      rmsNormEps,
      ropeTheta,
      ropeThetaSWA,
      headSizeFull,
      headSizeSWA,
      slidingWindow,
      logitSoftcapping,
      isSWA,
      nLayerKvFromStart,
      embeddingLengthPerLayer,
      expertCount,
      expertUsedCount,
      expertFeedForwardLength
    )

    if (!loadWeightsFlag) {
      return Llama(config, tokenizer, null)
    }

    val tensorEntries = GGUF.loadTensors(
      ggufPath,
      gguf.tensorDataOffset,
      tensorInfos
    )
    val qw = loadWeights(tensorEntries, config)
    return Llama(config, tokenizer, qw)
  }

  fun loadWeights(tensorEntries: Map<String, GGMLTensorEntry>, config: LlamaConfiguration): LlamaWeights {
    val ropeFreqsSWA = RoPE.precomputeFreqsCis(config.contextLength, config.headSizeSWA, config.ropeThetaSWA.toDouble())
    val ropeFreqsBuf = tensorEntries["rope_freqs.weight"]!!.toFloatBuffer()
    val modelRopeFreqs = FloatArray(ropeFreqsBuf.remaining())
    ropeFreqsBuf.get(modelRopeFreqs)
    val ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(
      config.contextLength,
      config.headSizeFull,
      config.ropeTheta.toDouble(),
      modelRopeFreqs
    )
    return loadWeightsWithRoPE(tensorEntries, config, ropeFreqsSWA, ropeFreqsFull)
  }

  fun loadWeightsWithRoPE(
    tensorEntries: Map<String, GGMLTensorEntry>,
    config: LlamaConfiguration,
    ropeFreqsSWA: Pair<FloatArray, FloatArray>,
    ropeFreqsFull: Pair<FloatArray, FloatArray>
  ): LlamaWeights {
    val numberOfLayers = config.numberOfLayers

    val tokenEmbeddingTable =
      loadQuantized(tensorEntries["token_embd.weight"]!!)

    // Load per-layer output scale (scalar per layer)
    val layerOutputScale = FloatArray(config.numberOfLayers)
    for (i in 0..<config.numberOfLayers) {
      val scaleEntry = tensorEntries["blk.$i.layer_output_scale.weight"]
      if (scaleEntry != null) {
        layerOutputScale[i] = scaleEntry.toFloatBuffer().get(0)
      } else {
        layerOutputScale[i] = 1.0f
      }
    }

    // Load per-layer embedding weights (if present)
    var perLayerTokenEmbd: FloatTensor? = null
    var perLayerModelProj: FloatTensor? = null
    var perLayerProjNorm: FloatBuffer? = null
    var perLayerInpGate: Array<FloatTensor>? = null
    var perLayerProj: Array<FloatTensor>? = null
    var perLayerPostNorm: Array<FloatBuffer>? = null

    if (config.embeddingLengthPerLayer > 0 && tensorEntries.containsKey("per_layer_token_embd.weight")) {
      perLayerTokenEmbd = loadQuantized(tensorEntries["per_layer_token_embd.weight"]!!)
      perLayerModelProj = loadQuantized(tensorEntries["per_layer_model_proj.weight"]!!)
      perLayerProjNorm = tensorEntries["per_layer_proj_norm.weight"]!!.toFloatBuffer()
      perLayerInpGate = loadArrayOfQuantized(config.numberOfLayers) { tensorEntries["blk.$it.inp_gate.weight"]!! }
      perLayerProj = loadArrayOfQuantized(config.numberOfLayers) { tensorEntries["blk.$it.proj.weight"]!! }
      perLayerPostNorm = loadArrayOfFloatBuffer(config.numberOfLayers) { tensorEntries["blk.$it.post_norm.weight"]!! }
    }

    // Load V weights (nullable: layers without V use K as V)
    val wv = arrayOfNulls<FloatTensor>(numberOfLayers)
    for (i in 0..<numberOfLayers) {
      val vEntry = tensorEntries["blk.$i.attn_v.weight"]
      wv[i] = if (vEntry != null) loadQuantized(vEntry) else null
    }

    // Load MoE weights (if present)
    var ffnGateInp: Array<FloatTensor>? = null
    var ffnGateInpScale: Array<FloatBuffer>? = null
    var ffnGateUpExps: Array<FloatTensor>? = null
    var ffnDownExps: Array<FloatTensor>? = null
    var ffnDownExpsScale: Array<FloatBuffer>? = null
    var ffnPostNorm1: Array<FloatBuffer>? = null
    var preFfwNorm2: Array<FloatBuffer>? = null
    var ffnPostNorm2: Array<FloatBuffer>? = null

    if (config.isMoE) {
      ffnGateInp = loadArrayOfQuantized(numberOfLayers) { tensorEntries["blk.$it.ffn_gate_inp.weight"]!! }
      ffnGateInpScale = loadArrayOfFloatBuffer(numberOfLayers) { tensorEntries["blk.$it.ffn_gate_inp.scale"]!! }
      ffnGateUpExps = loadArrayOfQuantized(numberOfLayers) { tensorEntries["blk.$it.ffn_gate_up_exps.weight"]!! }
      ffnDownExps = loadArrayOfQuantized(numberOfLayers) { tensorEntries["blk.$it.ffn_down_exps.weight"]!! }
      ffnDownExpsScale = loadArrayOfFloatBuffer(numberOfLayers) { tensorEntries["blk.$it.ffn_down_exps.scale"]!! }
      ffnPostNorm1 = loadArrayOfFloatBuffer(numberOfLayers) { tensorEntries["blk.$it.post_ffw_norm_1.weight"]!! }
      preFfwNorm2 = loadArrayOfFloatBuffer(numberOfLayers) { tensorEntries["blk.$it.pre_ffw_norm_2.weight"]!! }
      ffnPostNorm2 = loadArrayOfFloatBuffer(numberOfLayers) { tensorEntries["blk.$it.post_ffw_norm_2.weight"]!! }
    }

    return LlamaWeights(
      tokenEmbeddingTable,
      loadArrayOfFloatBuffer(config.numberOfLayers) { tensorEntries["blk.$it.attn_norm.weight"]!! },
      loadArrayOfQuantized(config.numberOfLayers) { tensorEntries["blk.$it.attn_q.weight"]!! },
      loadArrayOfQuantized(config.numberOfLayers) { tensorEntries["blk.$it.attn_k.weight"]!! },
      wv,
      loadArrayOfQuantized(config.numberOfLayers) { tensorEntries["blk.$it.attn_output.weight"]!! },
      loadArrayOfFloatBuffer(config.numberOfLayers) { tensorEntries["blk.$it.attn_q_norm.weight"]!! },
      loadArrayOfFloatBuffer(config.numberOfLayers) { tensorEntries["blk.$it.attn_k_norm.weight"]!! },
      loadArrayOfFloatBuffer(config.numberOfLayers) { tensorEntries["blk.$it.post_attention_norm.weight"]!! },
      loadArrayOfFloatBuffer(config.numberOfLayers) { tensorEntries["blk.$it.ffn_norm.weight"]!! },
      loadArrayOfQuantized(config.numberOfLayers) { tensorEntries["blk.$it.ffn_gate.weight"]!! },
      loadArrayOfQuantized(config.numberOfLayers) { tensorEntries["blk.$it.ffn_down.weight"]!! },
      loadArrayOfQuantized(config.numberOfLayers) { tensorEntries["blk.$it.ffn_up.weight"]!! },
      loadArrayOfFloatBuffer(config.numberOfLayers) { tensorEntries["blk.$it.post_ffw_norm.weight"]!! },
      tensorEntries["output_norm.weight"]!!.toFloatBuffer(),
      layerOutputScale,
      wrapWithFloatBuffer(ropeFreqsFull.first),
      wrapWithFloatBuffer(ropeFreqsFull.second),
      wrapWithFloatBuffer(ropeFreqsSWA.first),
      wrapWithFloatBuffer(ropeFreqsSWA.second),
      if (tensorEntries.containsKey("output.weight"))
        loadQuantized(tensorEntries["output.weight"]!!)
      else
        tokenEmbeddingTable,
      perLayerTokenEmbd, perLayerModelProj, perLayerProjNorm,
      perLayerInpGate, perLayerProj, perLayerPostNorm,
      ffnGateInp, ffnGateInpScale, ffnGateUpExps, ffnDownExps, ffnDownExpsScale,
      ffnPostNorm1, preFfwNorm2, ffnPostNorm2
    )
  }

  private fun createTokenizer(metadata: Map<String, Any>, vocabulary: Vocabulary): GemmaTokenizer {
    val tokenTypes = metadata["tokenizer.ggml.token_type"] as IntArray
    return GemmaTokenizer(vocabulary, tokenTypes)
  }

  fun loadQuantized(entry: GGMLTensorEntry): FloatTensor {
    val ggmlType = entry.ggmlType
    return FloatTensorFactory.create(ggmlType, entry)
  }

  fun loadArrayOfQuantized(size: Int, getTensorEntry: (Int) -> GGMLTensorEntry): Array<FloatTensor> {
    return Array(size) { loadQuantized(getTensorEntry(it)) }
  }

  fun loadArrayOfFloatBuffer(size: Int, getTensorEntry: (Int) -> GGMLTensorEntry): Array<FloatBuffer> {
    return Array(size) { getTensorEntry(it).toFloatBuffer() }
  }
}
