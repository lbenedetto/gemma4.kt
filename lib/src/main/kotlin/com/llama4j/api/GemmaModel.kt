package com.llama4j.api

import com.llama4j.api.Config.DEFAULT_MAX_TOKENS
import com.llama4j.internal.gguf.AOT
import com.llama4j.internal.gguf.ModelLoader
import com.llama4j.internal.model.*
import com.llama4j.internal.sampler.CategoricalSampler
import com.llama4j.internal.sampler.Sampler
import com.llama4j.internal.sampler.ToppSampler
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
class GemmaModel private constructor(private val model: Llama) {

  /**
   * Generate a response to [prompt].
   *
   * Each call is independent — no conversation state is retained.
   * For multi-turn exchanges use [chat] instead.
   */
  fun generate(prompt: String, configure: GenerationConfig.() -> Unit = {}): GenerationResult {
    val config = GenerationConfig().apply(configure)
    val state = model.createNewState()
    val chatFormat = GemmaChatFormat(model.tokenizer)
    val sampler = buildSampler(model.configuration.vocabularySize, config)

    val promptTokens = mutableListOf<Int>()
    if (config.thinking) {
      promptTokens.addAll(chatFormat.encodeSystemThinkingTurn(config.systemPrompt))
    } else if (config.systemPrompt != null) {
      promptTokens.addAll(chatFormat.encodeMessage(Message(Role.SYSTEM, config.systemPrompt!!)))
    }
    promptTokens.addAll(chatFormat.encodeMessage(Message(Role.USER, prompt)))
    promptTokens.addAll(chatFormat.encodeHeader(Message(Role.MODEL, "")))

    val stopTokens = chatFormat.stopTokens
    val (callback, buildResult) = tokenAccumulator(model, config)

    val responseTokens = Llama.generateTokens(
      model, state, 0, promptTokens, stopTokens,
      config.maxTokens, sampler, false, false, callback
    )
    if (responseTokens.isNotEmpty() && stopTokens.contains(responseTokens.last())) {
      responseTokens.removeLast()
    }

    return buildResult()
  }

  /**
   * Complete a fill-in-the-middle request given a [prefix] and [suffix].
   *
   * Requires a model that was trained with FIM support (e.g. Gemma 4 code variants).
   */
  fun fillInMiddle(prefix: String, suffix: String, configure: GenerationConfig.() -> Unit = {}): GenerationResult {
    val config = GenerationConfig().apply(configure)
    val state = model.createNewState()
    val chatFormat = GemmaChatFormat(model.tokenizer)
    val sampler = buildSampler(model.configuration.vocabularySize, config)
    val promptTokens = chatFormat.encodeFillInTheMiddle(prefix, suffix)
    val stopTokens = chatFormat.stopTokens
    val (callback, buildResult) = tokenAccumulator(model, config)

    val responseTokens = Llama.generateTokens(
      model, state, 0, promptTokens, stopTokens,
      config.maxTokens, sampler, false, false, callback
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
      inChannel = true; return@IntConsumer
    }
    if (hasThinkingSupport && token == channelClose) {
      inChannel = false; return@IntConsumer
    }
    if (tokenizer.isSpecialToken(token)) return@IntConsumer

    val piece = tokenizer.decode(listOf(token))
    if (inChannel) {
      thinkBuf?.append(piece)
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
