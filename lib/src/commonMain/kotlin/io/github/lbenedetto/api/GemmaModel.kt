package io.github.lbenedetto.api

expect class GemmaModel {
  /**
   * Generate a response to [prompt].
   *
   * Each call is independent — no conversation state is retained.
   * For multi-turn exchanges use [chat] instead.
   */
  fun generate(prompt: String, configure: GenerationConfig.() -> Unit = {}): GenerationResult

  /**
   * Complete a fill-in-the-middle request given a [prefix] and [suffix].
   *
   * Requires a model that was trained with FIM support (e.g. Gemma 4 code variants).
   */
  fun fillInMiddle(prefix: String, suffix: String, configure: GenerationConfig.() -> Unit = {}): GenerationResult

  /**
   * Create a new multi-turn [ChatSession] with the given configuration.
   *
   * The session maintains the full context of the conversation.
   * Call [ChatSession.reset] to start over.
   */
  fun chat(configure: GenerationConfig.() -> Unit = {}): ChatSession
}
