package io.github.lbenedetto.internal.gguf

import io.github.lbenedetto.api.Config.DEFAULT_MAX_TOKENS
import io.github.lbenedetto.internal.model.Llama
import io.github.lbenedetto.internal.util.Timer
import okio.FileSystem
import okio.Path
import okio.Path.Companion.toPath

internal object AOT {
  private val PRELOADED_GGUF = preLoadGGUF(System.getProperty("gemma4.PreloadGGUF"))

  private fun preLoadGGUF(modelPath: String?): PartialModel? {
    if (modelPath.isNullOrEmpty()) {
      return null
    }
    try {
      val path = modelPath.toPath()
      val metadata = FileSystem.SYSTEM.metadataOrNull(path)
      require(metadata != null && metadata.isRegularFile) { "Cannot pre-load model: $path" }
      val gguf: GGUF = GGUF.loadModel(path)
      val base = ModelLoader.loadModel(path, gguf, DEFAULT_MAX_TOKENS, false)
      // Precompute RoPE frequencies at build time (pure Java arrays, survives native-image)
      val config = base.configuration
      val ropeFreqsSWA = RoPE.precomputeFreqsCis(
        contextLength = config.contextLength,
        headSize = config.headSizeSWA,
        theta = config.ropeThetaSWA.toDouble()
      )
      // Read rope_freqs from model file
      val tmpEntries = GGUF.loadTensors(path, gguf.tensorDataOffset, gguf.tensorInfos)
      val ropeFreqsBuf = tmpEntries["rope_freqs.weight"]!!.toFloatBuffer()
      val modelRopeFreqs = FloatArray(ropeFreqsBuf.remaining())
      ropeFreqsBuf.get(modelRopeFreqs)
      val ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(
        contextLength = config.contextLength,
        headSize = config.headSizeFull,
        ropeTheta = config.ropeTheta.toDouble(),
        ropeFreqFactors = modelRopeFreqs
      )
      return PartialModel(
        modelFileName = path.name,
        model = base,
        tensorDataOffset = gguf.tensorDataOffset,
        tensorInfos = gguf.tensorInfos,
        ropeFreqsSWA = ropeFreqsSWA,
        ropeFreqsFull = ropeFreqsFull
      )
    } catch (e: Exception) {
      throw RuntimeException(e)
    }
  }

  fun tryUsePreLoaded(modelPath: Path, contextLength: Int): Llama? {
    val preLoaded = PRELOADED_GGUF ?: return null
    val optionsModel = modelPath.name
    val preLoadedModel = preLoaded.modelFileName
    if (optionsModel != preLoadedModel) {
      return null
    }
    val baseModel = preLoaded.model
    Timer.log("Load tensors from pre-loaded model").use {
      val tensorEntries: Map<String, GGMLTensorEntry> =
        GGUF.loadTensors(modelPath, preLoaded.tensorDataOffset, preLoaded.tensorInfos)
      val weights = ModelLoader.loadWeightsWithRoPE(
        tensorEntries, baseModel.configuration,
        preLoaded.ropeFreqsSWA, preLoaded.ropeFreqsFull
      )
      return Llama(baseModel.configuration.withContextLength(contextLength), baseModel.tokenizer, weights)
    }
  }
}
