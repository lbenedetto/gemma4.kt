package io.github.lbenedetto.api

import io.github.lbenedetto.internal.model.Llama

/**
 * Builds a token callback and a deferred result supplier that work together.
 *
 * The callback accumulates decoded text during generation, routing thinking-channel
 * tokens into a separate buffer when [GenerationConfig.thinking] is enabled.
 * The supplier returns the final [GenerationResult] once generation is complete.
 */
internal fun tokenAccumulator(
  model: Llama,
  config: GenerationConfig,
): Pair<(Int) -> Unit, () -> GenerationResult> {
  val tokenizer = model.tokenizer
  val channelOpen = tokenizer.specialTokens["<|channel>"]
  val channelClose = tokenizer.specialTokens["<channel|>"]
  val hasThinkingSupport = channelOpen != null && channelClose != null

  val textBuf = StringBuilder()
  val thinkBuf = if (config.thinking && hasThinkingSupport) StringBuilder() else null
  var inChannel = false

  val callback = consumer@{ token: Int ->
    if (hasThinkingSupport && token == channelOpen) {
      inChannel = true
      config.onThinkingStart?.invoke()
      return@consumer
    }
    if (hasThinkingSupport && token == channelClose) {
      inChannel = false
      config.onThinkingEnd?.invoke()
      return@consumer
    }
    if (tokenizer.isSpecialToken(token)) return@consumer

    val piece = tokenizer.decode(listOf(token))
    if (inChannel) {
      thinkBuf?.append(piece)
      config.onThinkingToken?.invoke(piece)
    } else {
      textBuf.append(piece)
      config.onToken?.invoke(piece)
    }
  }

  val buildResult: () -> GenerationResult = {
    GenerationResult(
      text = textBuf.toString(),
      thinking = thinkBuf?.toString()?.takeIf { it.isNotEmpty() },
    )
  }

  return callback to buildResult
}
