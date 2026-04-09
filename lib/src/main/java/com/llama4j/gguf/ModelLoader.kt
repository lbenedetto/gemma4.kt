package com.llama4j.gguf

import com.llama4j.floattensor.FloatTensor
import com.llama4j.floattensor.FloatTensorFactory
import com.llama4j.model.Llama
import com.llama4j.model.LlamaConfiguration
import com.llama4j.model.LlamaWeights
import com.llama4j.model.RoPE
import com.llama4j.tokenizer.GemmaTokenizer
import com.llama4j.tokenizer.Vocabulary
import java.io.IOException
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.util.*
import java.util.function.IntFunction

object ModelLoader {
  private fun loadVocabulary(metadata: MutableMap<String, Any>): Vocabulary {
    val tokens = Objects.requireNonNull<Array<String>>(metadata.get("tokenizer.ggml.tokens") as Array<String?>)
    val scores = Objects.requireNonNull<FloatArray>(metadata.get("tokenizer.ggml.scores") as FloatArray)
    return Vocabulary(tokens, scores)
  }

  @Throws(IOException::class)
  fun loadModel(ggufPath: Path, contextLength: Int): Llama {
    log("Load Gemma4 model").use { ignored ->
      FileChannel.open(ggufPath, StandardOpenOption.READ).use { fileChannel ->
        val gguf: GGUF = GGUF.Companion.loadModel(fileChannel, ggufPath.toString())
        return loadModel(fileChannel, gguf, contextLength, true)
      }
    }
  }

  @Throws(IOException::class)
  fun loadModel(fileChannel: FileChannel?, gguf: GGUF, contextLength: Int, loadWeightsFlag: Boolean): Llama {
    var contextLength = contextLength
    val metadata = gguf.getMetadata()

    val vocabulary = loadVocabulary(metadata)
    val tokenizer = createTokenizer(metadata, vocabulary)

    val modelContextLength = Objects.requireNonNull<Any>(metadata.get("gemma4.context_length")) as Int
    if (contextLength < 0 || modelContextLength < contextLength) {
      contextLength = modelContextLength
    }

    val embeddingLength = Objects.requireNonNull<Any>(metadata.get("gemma4.embedding_length")) as Int
    val numberOfHeads = Objects.requireNonNull<Any>(metadata.get("gemma4.attention.head_count")) as Int
    val numberOfLayers = Objects.requireNonNull<Any>(metadata.get("gemma4.block_count")) as Int

    val headSizeFull = Objects.requireNonNull<Any>(metadata.get("gemma4.attention.key_length")) as Int
    val headSizeSWA = Objects.requireNonNull<Any>(metadata.get("gemma4.attention.key_length_swa")) as Int
    val slidingWindow = Objects.requireNonNull<Any>(metadata.get("gemma4.attention.sliding_window")) as Int
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
    val ffnRaw = metadata.get("gemma4.feed_forward_length")
    if (ffnRaw is IntArray) {
      feedForwardLength = ffnRaw
    } else {
      feedForwardLength = IntArray(numberOfLayers)
      Arrays.fill(feedForwardLength, Objects.requireNonNull<Any>(ffnRaw) as Int)
    }

    val tensorInfos = gguf.getTensorInfos()

    // Derive isSWA per layer from Q norm weight size (256 = SWA, 512 = full attention)
    val isSWA: BooleanArray
    val swaRaw = metadata.get("gemma4.attention.sliding_window_pattern")
    if (swaRaw is BooleanArray) {
      isSWA = swaRaw
    } else {
      // Derive from tensor shapes: check Q norm size per layer
      isSWA = BooleanArray(numberOfLayers)
      for (i in 0..<numberOfLayers) {
        val qNorm = tensorInfos.get("blk." + i + ".attn_q_norm.weight")
        if (qNorm != null) {
          val qNormSize: Long = FloatTensor.Companion.numberOfElementsLong(*qNorm.dimensions)
          isSWA[i] = (qNormSize == headSizeSWA.toLong())
        } else {
          isSWA[i] = (i % 5 != 4) // fallback
        }
      }
    }

    // Derive per-layer KV head count from K weight shapes
    val numberOfKeyValueHeadsPerLayer = IntArray(numberOfLayers)
    for (i in 0..<numberOfLayers) {
      val kWeight = tensorInfos.get("blk." + i + ".attn_k.weight")
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

    val tensorEntries: MutableMap<String, GGMLTensorEntry> = GGUF.Companion.loadTensors(
      Objects.requireNonNull<FileChannel?>(fileChannel),
      gguf.getTensorDataOffset(),
      tensorInfos
    )
    val qw = loadWeights(tensorEntries, config)
    return Llama(config, tokenizer, qw)
  }

  fun loadWeights(tensorEntries: MutableMap<String, GGMLTensorEntry>, config: LlamaConfiguration): LlamaWeights {
    val ropeFreqsSWA = RoPE.precomputeFreqsCis(config.contextLength, config.headSizeSWA, config.ropeThetaSWA.toDouble())
    val ropeFreqsBuf = toFloatBuffer(Objects.requireNonNull<GGMLTensorEntry>(tensorEntries.get("rope_freqs.weight")))
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
    tensorEntries: MutableMap<String, GGMLTensorEntry>, config: LlamaConfiguration,
    ropeFreqsSWA: Pair<FloatArray, FloatArray>, ropeFreqsFull: Pair<FloatArray, FloatArray>
  ): LlamaWeights {
    val numberOfLayers = config.numberOfLayers

    val tokenEmbeddingTable =
      loadQuantized(Objects.requireNonNull<GGMLTensorEntry>(tensorEntries.get("token_embd.weight")))

    // Load per-layer output scale (scalar per layer)
    val layerOutputScale = FloatArray(config.numberOfLayers)
    for (i in 0..<config.numberOfLayers) {
      val scaleEntry = tensorEntries.get("blk." + i + ".layer_output_scale.weight")
      if (scaleEntry != null) {
        layerOutputScale[i] = toFloatBuffer(scaleEntry).get(0)
      } else {
        layerOutputScale[i] = 1.0f
      }
    }

    // Load per-layer embedding weights (if present)
    var perLayerTokenEmbd: FloatTensor? = null
    var perLayerModelProj: FloatTensor? = null
    var perLayerProjNorm: FloatBuffer? = null
    var perLayerInpGate: Array<FloatTensor> = null
    var perLayerProj: Array<FloatTensor> = null
    var perLayerPostNorm: Array<FloatBuffer> = null

    if (config.embeddingLengthPerLayer > 0 && tensorEntries.containsKey("per_layer_token_embd.weight")) {
      perLayerTokenEmbd =
        loadQuantized(Objects.requireNonNull<GGMLTensorEntry>(tensorEntries.get("per_layer_token_embd.weight")))
      perLayerModelProj =
        loadQuantized(Objects.requireNonNull<GGMLTensorEntry>(tensorEntries.get("per_layer_model_proj.weight")))
      perLayerProjNorm =
        toFloatBuffer(Objects.requireNonNull<GGMLTensorEntry>(tensorEntries.get("per_layer_proj_norm.weight")))
      perLayerInpGate = loadArrayOfQuantized(
        config.numberOfLayers,
        IntFunction { i: Int -> Objects.requireNonNull<GGMLTensorEntry>(tensorEntries.get("blk." + i + ".inp_gate.weight")) })
      perLayerProj = loadArrayOfQuantized(
        config.numberOfLayers,
        IntFunction { i: Int -> Objects.requireNonNull<GGMLTensorEntry>(tensorEntries.get("blk." + i + ".proj.weight")) })
      perLayerPostNorm = loadArrayOfFloatBuffer(
        config.numberOfLayers,
        IntFunction { i: Int -> Objects.requireNonNull<GGMLTensorEntry>(tensorEntries.get("blk." + i + ".post_norm.weight")) })
    }

    // Load V weights (nullable: layers without V use K as V)
    val wv = arrayOfNulls<FloatTensor>(numberOfLayers)
    for (i in 0..<numberOfLayers) {
      val vEntry = tensorEntries.get("blk." + i + ".attn_v.weight")
      wv[i] = if (vEntry != null) loadQuantized(vEntry) else null
    }

    // Load MoE weights (if present)
    var ffnGateInp: Array<FloatTensor> = null
    var ffnGateInpScale: Array<FloatBuffer> = null
    var ffnGateUpExps: Array<FloatTensor> = null
    var ffnDownExps: Array<FloatTensor> = null
    var ffnDownExpsScale: Array<FloatBuffer> = null
    var ffnPostNorm1: Array<FloatBuffer> = null
    var preFfwNorm2: Array<FloatBuffer> = null
    var ffnPostNorm2: Array<FloatBuffer> = null

    if (config.isMoE()) {
      ffnGateInp = loadArrayOfQuantized(
        numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".ffn_gate_inp.weight") })
      ffnGateInpScale = loadArrayOfFloatBuffer(
        numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".ffn_gate_inp.scale") })
      ffnGateUpExps = loadArrayOfQuantized(
        numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".ffn_gate_up_exps.weight") })
      ffnDownExps = loadArrayOfQuantized(
        numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".ffn_down_exps.weight") })
      ffnDownExpsScale = loadArrayOfFloatBuffer(
        numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".ffn_down_exps.scale") })
      ffnPostNorm1 = loadArrayOfFloatBuffer(
        numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".post_ffw_norm_1.weight") })
      preFfwNorm2 = loadArrayOfFloatBuffer(
        numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".pre_ffw_norm_2.weight") })
      ffnPostNorm2 = loadArrayOfFloatBuffer(
        numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".post_ffw_norm_2.weight") })
    }

    return LlamaWeights(
      tokenEmbeddingTable,
      loadArrayOfFloatBuffer(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".attn_norm.weight") }),
      loadArrayOfQuantized(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".attn_q.weight") }),
      loadArrayOfQuantized(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".attn_k.weight") }),
      wv,
      loadArrayOfQuantized(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".attn_output.weight") }),
      loadArrayOfFloatBuffer(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".attn_q_norm.weight") }),
      loadArrayOfFloatBuffer(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".attn_k_norm.weight") }),
      loadArrayOfFloatBuffer(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".post_attention_norm.weight") }),
      loadArrayOfFloatBuffer(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".ffn_norm.weight") }),
      loadArrayOfQuantized(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".ffn_gate.weight") }),
      loadArrayOfQuantized(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".ffn_down.weight") }),
      loadArrayOfQuantized(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".ffn_up.weight") }),
      loadArrayOfFloatBuffer(
        config.numberOfLayers,
        IntFunction { i: Int -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight") }),
      toFloatBuffer(Objects.requireNonNull<GGMLTensorEntry>(tensorEntries.get("output_norm.weight"))),
      layerOutputScale,
      FloatBuffer.wrap(ropeFreqsFull.first),
      FloatBuffer.wrap(ropeFreqsFull.second),
      FloatBuffer.wrap(ropeFreqsSWA.first),
      FloatBuffer.wrap(ropeFreqsSWA.second),
      if (tensorEntries.containsKey("output.weight"))
        ModelLoader.loadQuantized(tensorEntries.get("output.weight")!!)
      else
        tokenEmbeddingTable,
      perLayerTokenEmbd, perLayerModelProj, perLayerProjNorm,
      perLayerInpGate, perLayerProj, perLayerPostNorm,
      ffnGateInp, ffnGateInpScale, ffnGateUpExps, ffnDownExps, ffnDownExpsScale,
      ffnPostNorm1, preFfwNorm2, ffnPostNorm2
    )
  }

  private fun createTokenizer(metadata: MutableMap<String, Any>, vocabulary: Vocabulary): GemmaTokenizer {
    val tokenTypes = Objects.requireNonNull<Any>(metadata.get("tokenizer.ggml.token_type")) as IntArray
    return GemmaTokenizer(vocabulary, tokenTypes)
  }

  fun loadQuantized(entry: GGMLTensorEntry): FloatTensor {
    val ggmlType = entry.ggmlType
    return FloatTensorFactory.create(ggmlType, entry)
  }

  fun loadArrayOfQuantized(size: Int, getTensorEntry: IntFunction<GGMLTensorEntry>): Array<FloatTensor> {
    val array: Array<FloatTensor> = arrayOfNulls<FloatTensor>(size)
    for (i in 0..<size) {
      array[i] = loadQuantized(getTensorEntry.apply(i))
    }
    return array
  }

  fun loadArrayOfFloatBuffer(size: Int, getTensorEntry: IntFunction<GGMLTensorEntry>): Array<FloatBuffer> {
    val array: Array<FloatBuffer> = arrayOfNulls<FloatBuffer>(size)
    for (i in 0..<size) {
      array[i] = toFloatBuffer(getTensorEntry.apply(i))
    }
    return array
  }

  fun toFloatBuffer(tensorEntry: GGMLTensorEntry): FloatBuffer {
    val ggmlType = tensorEntry.ggmlType
    return when (ggmlType) {
      GGMLType.F32 -> tensorEntry.memorySegment.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
      else -> throw UnsupportedOperationException("Conversion to " + ggmlType)
    }
  }
}
