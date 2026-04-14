package io.github.lbenedetto.api

import kotlinx.cinterop.*
import llama.*

@OptIn(ExperimentalForeignApi::class)
actual class ChatSession internal constructor(
    private val model: CPointer<llama_model>,
    private val vocab: CPointer<llama_vocab>,
    private val chatTemplate: String?,
    private val ctx: CPointer<llama_context>,
    private val config: GenerationConfig,
    private val maxContext: Int,
) {
    private val messages = mutableListOf<Pair<String, String>>()
    private var tokenCount = 0
    private var prevFormatted = ""

    init {
        config.systemPrompt?.let {
            messages.add("system" to applyThinkingTag(it, config))
            prevFormatted = formatMessages(chatTemplate, messages, addAssistant = false)
        }
    }

    actual val contextUsed: Int get() = tokenCount
    actual val contextRemaining: Int get() = maxContext - tokenCount

    actual fun send(message: String): GenerationResult {
        messages.add("user" to message)

        // Format full conversation so far and extract only the new part
        val fullFormatted = formatMessages(chatTemplate, messages, addAssistant = true)
        val newPart = fullFormatted.removePrefix(prevFormatted)

        // Tokenize only the incremental part (no BOS — already in context or first turn handles it)
        val addBos = tokenCount == 0
        val newTokens = tokenize(newPart, addBos)

        // Decode new prompt tokens
        newTokens.usePinned { pinned ->
            val batch = llama_batch_get_one(pinned.addressOf(0), newTokens.size)
            val rc = llama_decode(ctx, batch.readValue())
            check(rc == 0) { "llama_decode failed: $rc" }
        }
        tokenCount += newTokens.size

        // Build sampler
        val sparams = llama_sampler_chain_default_params()
        val sampler = llama_sampler_chain_init(sparams.readValue())!!
        if (config.temperature == 0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temperature))
            if (config.topP in 0f..1f) {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config.topP, 1u))
            }
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(config.seed.toUInt()))
        }

        val textBuf = StringBuilder()
        val thinkBuf = if (config.thinking) StringBuilder() else null
        var inThinking = false
        val channelOpen = "<|channel>"
        val channelClose = "<channel|>"

        val maxGenerate = maxContext - tokenCount

        try {
            for (i in 0 until maxGenerate) {
                val token = llama_sampler_sample(sampler, ctx, -1)
                llama_sampler_accept(sampler, token)

                if (llama_vocab_is_eog(vocab, token)) {
                    tokenCount++ // count the stop token
                    break
                }

                val piece = tokenToString(vocab, token)
                tokenCount++

                if (config.thinking) {
                    if (piece.contains(channelOpen)) {
                        inThinking = true
                        config.onThinkingStart?.invoke()
                    } else if (piece.contains(channelClose)) {
                        inThinking = false
                        config.onThinkingEnd?.invoke()
                    } else if (inThinking) {
                        thinkBuf?.append(piece)
                        config.onThinkingToken?.invoke(piece)
                    } else {
                        textBuf.append(piece)
                        config.onToken?.invoke(piece)
                    }
                } else {
                    textBuf.append(piece)
                    config.onToken?.invoke(piece)
                }

                // Feed token back
                memScoped {
                    val tokenArr = allocArray<IntVar>(1)
                    tokenArr[0] = token
                    val batch = llama_batch_get_one(tokenArr, 1)
                    val rc = llama_decode(ctx, batch.readValue())
                    if (rc != 0) break
                }
            }
        } finally {
            llama_sampler_free(sampler)
        }

        val responseText = textBuf.toString()
        messages.add("model" to responseText)
        prevFormatted = formatMessages(chatTemplate, messages, addAssistant = false)

        return GenerationResult(
            text = responseText,
            thinking = thinkBuf?.toString()?.takeIf { it.isNotEmpty() },
        )
    }

    actual fun reset() {
        messages.clear()
        tokenCount = 0
        prevFormatted = ""
        val mem = llama_get_memory(ctx)
        if (mem != null) {
            llama_memory_clear(mem, true)
        }
        // Re-add system prompt if present
        config.systemPrompt?.let {
            messages.add("system" to applyThinkingTag(it, config))
            prevFormatted = formatMessages(chatTemplate, messages, addAssistant = false)
        }
    }

    private fun tokenize(text: String, addBos: Boolean): IntArray {
        val maxTokens = text.length + 16
        val buf = IntArray(maxTokens)
        val n = buf.usePinned { pinned ->
            llama_tokenize(vocab, text, text.length, pinned.addressOf(0), maxTokens, addBos, true)
        }
        check(n >= 0) { "Tokenization failed" }
        return buf.copyOf(n)
    }
}