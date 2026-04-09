package com.llama4j.gguf

import com.llama4j.Options
import com.llama4j.model.Llama
import com.llama4j.model.RoPE
import org.jetbrains.annotations.Contract
import java.io.IOException
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.util.*

object AOT {
  private val PRELOADED_GGUF = preLoadGGUF(System.getProperty("gemma4.PreloadGGUF"))

  @Contract("null -> null")
  private fun preLoadGGUF(modelPath: String?): PartialModel? {
    if (modelPath == null || modelPath.isEmpty()) {
      return null
    }
    try {
      val path = Path.of(modelPath)
      require(!(!Files.exists(path) || !Files.isRegularFile(path))) { "Cannot pre-load model: " + path }
      FileChannel.open(path, StandardOpenOption.READ).use { fileChannel ->
        val gguf: GGUF = GGUF.Companion.loadModel(fileChannel, path.toString())
        val base = ModelLoader.loadModel(null, gguf, Options.Companion.DEFAULT_MAX_TOKENS, false)
        // Precompute RoPE frequencies at build time (pure Java arrays, survives native-image)
        val config = base.configuration
        val ropeFreqsSWA = RoPE.precomputeFreqsCis(
          config.contextLength, config.headSizeSWA, config.ropeThetaSWA.toDouble()
        )
        // Read rope_freqs from model file
        val ropeFreqsFull: Pair<FloatArray, FloatArray>?
        val tmpEntries: MutableMap<String, GGMLTensorEntry> =
          GGUF.Companion.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos())
        val ropeFreqsBuf =
          ModelLoader.toFloatBuffer(Objects.requireNonNull<GGMLTensorEntry>(tmpEntries.get("rope_freqs.weight")))
        val modelRopeFreqs = FloatArray(ropeFreqsBuf.remaining())
        ropeFreqsBuf.get(modelRopeFreqs)
        ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(
          config.contextLength, config.headSizeFull, config.ropeTheta.toDouble(), modelRopeFreqs
        )
        return PartialModel(
          path.getFileName().toString(), base,
          gguf.getTensorDataOffset(), gguf.getTensorInfos(),
          ropeFreqsSWA, ropeFreqsFull
        )
      }
    } catch (e: IOException) {
      throw RuntimeException(e)
    }
  }

  @Throws(IOException::class)
  fun tryUsePreLoaded(modelPath: Path, contextLength: Int): Llama? {
    val preLoaded = PRELOADED_GGUF
    if (preLoaded == null) {
      return null
    }
    val optionsModel = modelPath.getFileName().toString()
    val preLoadedModel = preLoaded.modelFileName
    if (optionsModel != preLoadedModel) {
      return null
    }
    val baseModel = preLoaded.model
    log("Load tensors from pre-loaded model").use { timer ->
      FileChannel.open(modelPath, StandardOpenOption.READ).use { fileChannel ->
        val tensorEntries: MutableMap<String, GGMLTensorEntry> =
          GGUF.Companion.loadTensors(fileChannel, preLoaded.tensorDataOffset, preLoaded.tensorInfos)
        val weights = ModelLoader.loadWeightsWithRoPE(
          tensorEntries, baseModel.configuration,
          preLoaded.ropeFreqsSWA, preLoaded.ropeFreqsFull
        )
        return Llama(baseModel.configuration.withContextLength(contextLength), baseModel.tokenizer, weights)
      }
    }
  }
}
