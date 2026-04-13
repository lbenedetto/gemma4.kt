package io.github.lbenedetto.api

import io.github.lbenedetto.api.Config.DEFAULT_MAX_TOKENS
import io.github.lbenedetto.internal.gguf.AOT
import io.github.lbenedetto.internal.gguf.ModelLoader
import io.github.lbenedetto.internal.model.*
import io.github.lbenedetto.internal.sampler.CategoricalSampler
import io.github.lbenedetto.internal.sampler.Sampler
import io.github.lbenedetto.internal.sampler.ToppSampler
import java.nio.file.Path
import java.util.function.IntConsumer
import java.util.random.RandomGeneratorFactory

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
actual class GemmaModel private constructor(private val model: Llama) {

  actual fun generate(prompt: String, configure: GenerationConfig.() -> Unit): GenerationResult =
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

  actual fun fillInMiddle(prefix: String, suffix: String, configure: GenerationConfig.() -> Unit): GenerationResult =
    doGenerate(configure) { _, chatFormat -> chatFormat.encodeFillInTheMiddle(prefix, suffix) }

  private fun doGenerate(
    configure: GenerationConfig.() -> Unit,
    buildPromptTokens: (GenerationConfig, GemmaChatFormat) -> List<Int>,
  ): GenerationResult {
    val config = GenerationConfig().apply(configure)
    val chatFormat = GemmaChatFormat(model.tokenizer)
    val state = model.createNewState()
    val sampler = buildSampler(model.configuration.vocabularySize, config)
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

  actual fun chat(configure: GenerationConfig.() -> Unit): ChatSession {
    return ChatSession(model, GenerationConfig().apply(configure))
  }

  companion object {
    /**
     * Load a model from a GGUF file. This is the main entry point.
     *
     * @param modelPath Path to the `.gguf` model file.
     * @param contextLength Maximum context length. < 0 uses the model's built-in default of [DEFAULT_MAX_TOKENS]
     */
    @JvmStatic
    @JvmOverloads
    fun load(modelPath: Path, contextLength: Int = DEFAULT_MAX_TOKENS): GemmaModel {
      val llama = AOT.tryUsePreLoaded(modelPath, contextLength)
        ?: ModelLoader.loadModel(modelPath, contextLength)
      return GemmaModel(llama)
    }

    internal fun buildSampler(vocabularySize: Int, config: GenerationConfig): Sampler {
      if (config.temperature == 0.0f) return Sampler.ARGMAX
      val rng = RandomGeneratorFactory.getDefault().create(config.seed)
      val inner: Sampler = if (config.topP <= 0f || config.topP >= 1f) {
        CategoricalSampler(rng)
      } else {
        ToppSampler(vocabularySize, config.topP, rng)
      }
      return Sampler { logits ->
        val size = Math.toIntExact(logits.size)
        logits.divideInPlace(0, size, config.temperature)
        logits.softmaxInPlace(0, size)
        inner.sampleToken(logits)
      }
    }
  }
}

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
): Pair<IntConsumer, () -> GenerationResult> {
  val tokenizer = model.tokenizer
  val channelOpen = tokenizer.specialTokens["<|channel>"]
  val channelClose = tokenizer.specialTokens["<channel|>"]
  val hasThinkingSupport = channelOpen != null && channelClose != null

  val textBuf = StringBuilder()
  val thinkBuf = if (config.thinking && hasThinkingSupport) StringBuilder() else null
  var inChannel = false

  val callback = IntConsumer { token ->
    if (hasThinkingSupport && token == channelOpen) {
      inChannel = true
      config.onThinkingStart?.invoke()
      return@IntConsumer
    }
    if (hasThinkingSupport && token == channelClose) {
      inChannel = false
      config.onThinkingEnd?.invoke()
      return@IntConsumer
    }
    if (tokenizer.isSpecialToken(token)) return@IntConsumer

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
