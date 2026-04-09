package com.llama4j.gguf

import com.llama4j.Config.DEFAULT_MAX_TOKENS
import com.llama4j.model.Llama
import com.llama4j.model.RoPE
import com.llama4j.util.Timer
import org.jetbrains.annotations.Contract
import java.io.IOException
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption

object AOT {
  private val PRELOADED_GGUF = preLoadGGUF(System.getProperty("gemma4.PreloadGGUF"))

  @Contract("null -> null")
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
          config.contextLength, config.headSizeSWA, config.ropeThetaSWA.toDouble()
        )
        // Read rope_freqs from model file
        val ropeFreqsFull: Pair<FloatArray, FloatArray>?
        val tmpEntries: MutableMap<String, GGMLTensorEntry> =
          GGUF.loadTensors(fileChannel, gguf.tensorDataOffset, gguf.tensorInfos)
        val ropeFreqsBuf =
          ModelLoader.toFloatBuffer(tmpEntries["rope_freqs.weight"]!!)
        val modelRopeFreqs = FloatArray(ropeFreqsBuf.remaining())
        ropeFreqsBuf.get(modelRopeFreqs)
        ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(
          config.contextLength, config.headSizeFull, config.ropeTheta.toDouble(), modelRopeFreqs
        )
        return PartialModel(
          path.fileName.toString(), base,
          gguf.tensorDataOffset, gguf.tensorInfos,
          ropeFreqsSWA, ropeFreqsFull
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
        val tensorEntries: MutableMap<String, GGMLTensorEntry> =
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
