package io.github.lbenedetto.api

import io.github.lbenedetto.internal.gguf.ModelLoader
import okio.Path

actual object GemmaModel {
  actual fun load(modelPath: Path, contextLength: Int): Gemma {
    val llama = ModelLoader.loadModel(modelPath, contextLength)
    return Gemma(llama)
  }
}
