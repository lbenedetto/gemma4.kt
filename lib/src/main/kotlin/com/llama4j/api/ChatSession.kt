package com.llama4j.api

import com.llama4j.internal.model.*

/**
 * A stateful multi-turn conversation. Obtain one from [GemmaModel.chat].
 *
 * The session owns its [LlamaState] and accumulates conversation tokens across
 * calls to [send], so the model always has full context of the prior exchange.
 */
class ChatSession internal constructor(
    private val model: Llama,
    private val config: GenerationConfig,
) {
    private var state: LlamaState? = null
    private val chatFormat = GemmaChatFormat(model.tokenizer)
    private val conversationTokens = mutableListOf<Int>()
    private var startPosition = 0

    init {
        if (config.thinking) {
            conversationTokens.addAll(chatFormat.encodeSystemThinkingTurn(config.systemPrompt))
        } else if (config.systemPrompt != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(Message(Role.SYSTEM, config.systemPrompt!!)))
        }
    }

    /** How many context tokens have been consumed so far. */
    val contextUsed: Int get() = conversationTokens.size

    /** How many context tokens remain before the window is full. */
    val contextRemaining: Int get() = config.maxTokens - conversationTokens.size

    /** Send [message] and return the model's response. */
    fun send(message: String): GenerationResult {
        if (state == null) state = model.createNewState()

        conversationTokens.addAll(chatFormat.encodeMessage(Message(Role.USER, message)))
        conversationTokens.addAll(chatFormat.encodeHeader(Message(Role.MODEL, "")))

        val sampler = GemmaModel.buildSampler(model.configuration.vocabularySize, config)
        val stopTokens = chatFormat.stopTokens
        val (callback, buildResult) = tokenAccumulator(model, config)

        val responseTokens = Llama.generateTokens(
            model, state!!, startPosition,
            conversationTokens.subList(startPosition, conversationTokens.size),
            stopTokens, config.maxTokens, sampler,
            false, false, callback
        )

        // Separate stop token from response body (same bookkeeping as the CLI)
        var stopToken: Int? = null
        if (responseTokens.isNotEmpty() && stopTokens.contains(responseTokens.last())) {
            stopToken = responseTokens.removeLast()
        }

        conversationTokens.addAll(responseTokens)
        if (stopToken != null) conversationTokens.add(stopToken)
        startPosition = conversationTokens.size

        return buildResult()
    }

    /**
     * Reset the conversation history. The system prompt and all other config settings
     * are preserved; only the accumulated turn history is cleared.
     */
    fun reset() {
        state = null
        conversationTokens.clear()
        startPosition = 0
        if (config.thinking) {
            conversationTokens.addAll(chatFormat.encodeSystemThinkingTurn(config.systemPrompt))
        } else if (config.systemPrompt != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(Message(Role.SYSTEM, config.systemPrompt!!)))
        }
    }
}
