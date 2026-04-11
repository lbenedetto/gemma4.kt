package io.github.lbenedetto.api

import okio.Path

expect object GemmaModel {

  /**
   * Load a model from a GGUF file. This is the main entry point.
   *
   * @param modelPath Path to the `.gguf` model file.
   * @param contextLength Maximum context length. < 0 uses the model's built-in default of [Config.DEFAULT_MAX_TOKENS]
   */
  fun load(modelPath: Path, contextLength: Int = Config.DEFAULT_MAX_TOKENS): Gemma
}
