package io.github.lbenedetto.api

actual class GemmaModel {
  actual fun generate(
    prompt: String,
    configure: GenerationConfig.() -> Unit
  ): GenerationResult {
    TODO("Not yet implemented")
  }

  actual fun fillInMiddle(
    prefix: String,
    suffix: String,
    configure: GenerationConfig.() -> Unit
  ): GenerationResult {
    TODO("Not yet implemented")
  }

  actual fun chat(configure: GenerationConfig.() -> Unit): ChatSession {
    TODO("Not yet implemented")
  }
}
