package io.github.lbenedetto.api

/**
 * Configuration for a generation request.
 *
 * Use the DSL block passed to [GemmaModel.generate], [GemmaModel.fillInMiddle], or [GemmaModel.chat]:
 * ```kotlin
 * model.generate("Tell me a joke") {
 *     temperature = 0.8f
 *     maxTokens = 256
 *     onToken = { piece -> print(piece) }
 * }
 * ```
 */
class GenerationConfig {
  /** Sampling temperature. 0 = greedy (argmax), higher = more random. Default: 1.0 */
  var temperature: Float = 1.0f

  /** Top-p nucleus sampling threshold. 0 or ≥1 disables it. Default: 0.95 */
  var topP: Float = 0.95f

  /** Random seed for reproducibility. Default: random. */
  var seed: Long = System.nanoTime()

  /** Maximum tokens to generate. Negative = capped by context length. Default: 1024 */
  var maxTokens: Int = Config.DEFAULT_MAX_TOKENS

  /** Optional system prompt prepended to the conversation. */
  var systemPrompt: String? = null

  /**
   * Enable thinking mode. When true, the model reasons internally before answering.
   * Thinking tokens are excluded from the streaming [onToken] callback and from
   * [GenerationResult.text], but available in [GenerationResult.thinking].
   */
  var thinking: Boolean = false

  /**
   * Streaming callback. Called with each decoded text piece as it is generated.
   * Thinking tokens are never included here, even when [thinking] is true.
   * Set to null (the default) to disable streaming.
   */
  var onToken: ((String) -> Unit)? = null

  /**
   * Called when the model enters a thinking channel (before the first thinking token).
   * Only fires when [thinking] is true and the model supports thinking.
   */
  var onThinkingStart: (() -> Unit)? = null

  /**
   * Streaming callback for thinking tokens. Called with each decoded thinking piece.
   * Only fires when [thinking] is true and the model supports thinking.
   */
  var onThinkingToken: ((String) -> Unit)? = null

  /**
   * Called when the model exits a thinking channel (after the last thinking token).
   * Only fires when [thinking] is true and the model supports thinking.
   */
  var onThinkingEnd: (() -> Unit)? = null
}
