package io.github.lbenedetto.api

import io.github.lbenedetto.internal.model.*
import io.github.lbenedetto.internal.sampler.Sampler

/**
 * A loaded Gemma model, ready for text generation.
 *
 * ### Quick start
 * ```kotlin
 * val model = GemmaModel.load(Path.of("gemma-4-E2B-it-Q8_0.gguf"))
 *
 * // Single prompt
 * println(model.generate("Why is the sky blue?").text)
 *
 * // With options and streaming
 * val result = model.generate("Write a haiku") {
 *     temperature = 0.7f
 *     onToken = { piece -> print(piece) }
 * }
 *
 * // Multi-turn chat
 * val chat = model.chat { systemPrompt = "You are a helpful assistant." }
 * println(chat.send("Hello!").text)
 * println(chat.send("What did I just say?").text)
 *
 * // Fill-in-the-middle (code completion)
 * println(model.fillInMiddle(prefix = "fun greet(name: String) = ", suffix = "").text)
 * ```
 */
class Gemma internal constructor(private val model: Llama) {
  /**
   * Generate a response to [prompt].
   *
   * Each call is independent — no conversation state is retained.
   * For multi-turn exchanges use [chat] instead.
   */
  fun generate(prompt: String, configure: GenerationConfig.() -> Unit = {}): GenerationResult =
    doGenerate(configure) { config, chatFormat ->
      buildList {
        if (config.thinking) {
          addAll(chatFormat.encodeSystemThinkingTurn(config.systemPrompt))
        } else if (config.systemPrompt != null) {
          addAll(chatFormat.encodeMessage(Message(Role.SYSTEM, config.systemPrompt!!)))
        }
        addAll(chatFormat.encodeMessage(Message(Role.USER, prompt)))
        addAll(chatFormat.encodeHeader(Message(Role.MODEL, "")))
      }
    }

  /**
   * Complete a fill-in-the-middle request given a [prefix] and [suffix].
   *
   * Requires a model that was trained with FIM support (e.g. Gemma 4 code variants).
   */
  fun fillInMiddle(prefix: String, suffix: String, configure: GenerationConfig.() -> Unit = {}): GenerationResult =
    doGenerate(configure) { _, chatFormat -> chatFormat.encodeFillInTheMiddle(prefix, suffix) }

  private fun doGenerate(
    configure: GenerationConfig.() -> Unit,
    buildPromptTokens: (GenerationConfig, GemmaChatFormat) -> List<Int>,
  ): GenerationResult {
    val config = GenerationConfig().apply(configure)
    val chatFormat = GemmaChatFormat(model.tokenizer)
    val state = model.createNewState()
    val sampler = Sampler.build(model.configuration.vocabularySize, config)
    val promptTokens = buildPromptTokens(config, chatFormat).toList()
    val stopTokens = chatFormat.stopTokens
    val (callback, buildResult) = tokenAccumulator(model, config)

    val responseTokens = Llama.generateTokens(
      model, state, 0, promptTokens, stopTokens,
      config.maxTokens, sampler, echo = false, color = false, callback
    )
    if (responseTokens.isNotEmpty() && stopTokens.contains(responseTokens.last())) {
      responseTokens.removeLast()
    }

    return buildResult()
  }

  /**
   * Create a new multi-turn [ChatSession] with the given configuration.
   *
   * The session maintains a [LlamaState] across turns so the model retains
   * full context of the conversation. Call [ChatSession.reset] to start over.
   */
  fun chat(configure: GenerationConfig.() -> Unit = {}): ChatSession {
    return ChatSession(model, GenerationConfig().apply(configure))
  }
}
