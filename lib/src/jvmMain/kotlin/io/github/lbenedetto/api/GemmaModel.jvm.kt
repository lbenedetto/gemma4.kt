package io.github.lbenedetto.api

import io.github.lbenedetto.api.Config.DEFAULT_MAX_TOKENS
import io.github.lbenedetto.internal.gguf.AOT
import io.github.lbenedetto.internal.gguf.ModelLoader
import okio.Path
import okio.Path.Companion.toPath

actual object GemmaModel {
  @JvmStatic
  actual fun load(modelPath: Path, contextLength: Int): Gemma {
    val llama = AOT.tryUsePreLoaded(modelPath, contextLength)
      ?: ModelLoader.loadModel(modelPath, contextLength)
    return Gemma(llama)
  }

  @JvmStatic
  @JvmOverloads
  fun load(modelPath: java.nio.file.Path, contextLength: Int = DEFAULT_MAX_TOKENS): Gemma =
    load(modelPath.toString().toPath(), contextLength)
}
