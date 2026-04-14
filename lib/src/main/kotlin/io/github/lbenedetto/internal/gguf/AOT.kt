package io.github.lbenedetto.internal.gguf

import io.github.lbenedetto.api.Config.DEFAULT_MAX_TOKENS
import io.github.lbenedetto.internal.model.Llama
import io.github.lbenedetto.internal.model.RoPE
import io.github.lbenedetto.internal.util.Timer
import java.io.IOException
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption

internal object AOT {
  private val PRELOADED_GGUF = preLoadGGUF(System.getProperty("gemma4.PreloadGGUF"))

  private fun preLoadGGUF(modelPath: String?): PartialModel? {
    if (modelPath.isNullOrEmpty()) {
      return null
    }
    try {
      val path = Path.of(modelPath)
      require(!(!Files.exists(path) || !Files.isRegularFile(path))) { "Cannot pre-load model: $path" }
      FileChannel.open(path, StandardOpenOption.READ).use { fileChannel ->
        val gguf: GGUF = GGUF.loadModel(fileChannel, path.toString())
        val base = ModelLoader.loadModel(null, gguf, DEFAULT_MAX_TOKENS, false)
        // Precompute RoPE frequencies at build time (pure Java arrays, survives native-image)
        val config = base.configuration
        val ropeFreqsSWA = RoPE.precomputeFreqsCis(
          contextLength = config.contextLength,
          headSize = config.headSizeSWA,
          theta = config.ropeThetaSWA.toDouble()
        )
        // Read rope_freqs from model file
        val tmpEntries = GGUF.loadTensors(fileChannel, gguf.tensorDataOffset, gguf.tensorInfos)
        val ropeFreqsBuf = ModelLoader.toFloatBuffer(tmpEntries["rope_freqs.weight"]!!)
        val modelRopeFreqs = FloatArray(ropeFreqsBuf.remaining())
        ropeFreqsBuf.get(modelRopeFreqs)
        val ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(
          contextLength = config.contextLength,
          headSize = config.headSizeFull,
          ropeTheta = config.ropeTheta.toDouble(),
          ropeFreqFactors = modelRopeFreqs
        )
        return PartialModel(
          modelFileName = path.fileName.toString(),
          model = base,
          tensorDataOffset = gguf.tensorDataOffset,
          tensorInfos = gguf.tensorInfos,
          ropeFreqsSWA = ropeFreqsSWA,
          ropeFreqsFull = ropeFreqsFull
        )
      }
    } catch (e: IOException) {
      throw RuntimeException(e)
    }
  }

  @Throws(IOException::class)
  fun tryUsePreLoaded(modelPath: Path, contextLength: Int): Llama? {
    val preLoaded = PRELOADED_GGUF ?: return null
    val optionsModel = modelPath.fileName.toString()
    val preLoadedModel = preLoaded.modelFileName
    if (optionsModel != preLoadedModel) {
      return null
    }
    val baseModel = preLoaded.model
    Timer.log("Load tensors from pre-loaded model").use {
      FileChannel.open(modelPath, StandardOpenOption.READ).use { fileChannel ->
        val tensorEntries: Map<String, GGMLTensorEntry> =
          GGUF.loadTensors(fileChannel, preLoaded.tensorDataOffset, preLoaded.tensorInfos)
        val weights = ModelLoader.loadWeightsWithRoPE(
          tensorEntries, baseModel.configuration,
          preLoaded.ropeFreqsSWA, preLoaded.ropeFreqsFull
        )
        return Llama(baseModel.configuration.withContextLength(contextLength), baseModel.tokenizer, weights)
      }
    }
  }
}
