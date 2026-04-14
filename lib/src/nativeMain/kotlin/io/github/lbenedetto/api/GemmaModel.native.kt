package io.github.lbenedetto.api

import kotlinx.cinterop.*
import llama.*

@OptIn(ExperimentalForeignApi::class)
actual class GemmaModel private constructor(
    private val model: CPointer<llama_model>,
    private val contextLength: Int,
) {
    internal val vocab: CPointer<llama_vocab> = llama_model_get_vocab(model)!!
    private val chatTemplate: String? = llama_model_chat_template(model, null)?.toKString()

    companion object {
        fun load(modelPath: String, contextLength: Int = Config.DEFAULT_MAX_TOKENS): GemmaModel {
            val params = llama_model_default_params()
            val m = llama_model_load_from_file(modelPath, params)
                ?: error("Failed to load model: $modelPath")
            return GemmaModel(m, contextLength)
        }
    }

    actual fun generate(prompt: String, configure: GenerationConfig.() -> Unit): GenerationResult {
        val config = GenerationConfig().apply(configure)
        val ctx = createContext(config)
        try {
            val formatted = formatMessages(
                chatTemplate,
                buildList {
                    config.systemPrompt?.let { add("system" to applyThinkingTag(it, config)) }
                    add("user" to prompt)
                },
                addAssistant = true
            )
            val tokens = tokenize(formatted, addBos = true)
            return runGeneration(ctx, tokens, config)
        } finally {
            llama_free(ctx)
        }
    }

    actual fun fillInMiddle(prefix: String, suffix: String, configure: GenerationConfig.() -> Unit): GenerationResult {
        val config = GenerationConfig().apply(configure)
        val ctx = createContext(config)
        try {
            val fimPrefix = llama_vocab_fim_pre(vocab)
            val fimSuffix = llama_vocab_fim_suf(vocab)
            val fimMiddle = llama_vocab_fim_mid(vocab)

            val prefixTokens = tokenize(prefix, addBos = false)
            val suffixTokens = tokenize(suffix, addBos = false)

            val tokens = IntArray(prefixTokens.size + suffixTokens.size + 3)
            var i = 0
            tokens[i++] = fimPrefix
            for (t in prefixTokens) tokens[i++] = t
            tokens[i++] = fimSuffix
            for (t in suffixTokens) tokens[i++] = t
            tokens[i++] = fimMiddle

            return runGeneration(ctx, tokens, config)
        } finally {
            llama_free(ctx)
        }
    }

    actual fun chat(configure: GenerationConfig.() -> Unit): ChatSession {
        val config = GenerationConfig().apply(configure)
        val ctx = createContext(config)
        return ChatSession(model, vocab, chatTemplate, ctx, config, contextLength)
    }

    internal fun createContext(config: GenerationConfig): CPointer<llama_context> {
        memScoped {
            val ctxSize = if (config.maxTokens > 0) config.maxTokens.toUInt() else contextLength.toUInt()
            val p = llama_context_default_params().getPointer(this)
            p.pointed.n_ctx = ctxSize
            p.pointed.n_batch = ctxSize
            return llama_init_from_model(model, p.pointed.readValue())
                ?: error("Failed to create llama context")
        }
    }

    internal fun tokenize(text: String, addBos: Boolean): IntArray {
        val maxTokens = text.length + 16
        val buf = IntArray(maxTokens)
        val n = buf.usePinned { pinned ->
            llama_tokenize(vocab, text, text.length, pinned.addressOf(0), maxTokens, addBos, true)
        }
        check(n >= 0) { "Tokenization failed" }
        return buf.copyOf(n)
    }

    private fun runGeneration(
        ctx: CPointer<llama_context>,
        promptTokens: IntArray,
        config: GenerationConfig,
    ): GenerationResult {
        val sampler = buildSampler(config)
        try {
            return decodeAndSample(ctx, vocab, sampler, promptTokens, config)
        } finally {
            llama_sampler_free(sampler)
        }
    }

    private fun buildSampler(config: GenerationConfig): CPointer<llama_sampler> {
        val sparams = llama_sampler_chain_default_params()
        val chain = llama_sampler_chain_init(sparams)!!
        if (config.temperature == 0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_temp(config.temperature))
            if (config.topP in 0f..1f) {
                llama_sampler_chain_add(chain, llama_sampler_init_top_p(config.topP, 1u))
            }
            llama_sampler_chain_add(chain, llama_sampler_init_dist(config.seed.toUInt()))
        }
        return chain
    }
}

@OptIn(ExperimentalForeignApi::class)
internal fun applyThinkingTag(systemPrompt: String, config: GenerationConfig): String {
    return if (config.thinking) "<think>\n$systemPrompt" else systemPrompt
}

@OptIn(ExperimentalForeignApi::class)
internal fun formatMessages(
    chatTemplate: String?,
    messages: List<Pair<String, String>>,
    addAssistant: Boolean,
): String {
    memScoped {
        val chatMessages = allocArray<llama_chat_message>(messages.size)
        val cStrings = messages.map { (role, content) -> role.cstr.ptr to content.cstr.ptr }
        for (i in messages.indices) {
            chatMessages[i].role = cStrings[i].first
            chatMessages[i].content = cStrings[i].second
        }

        var bufSize = 1024
        while (true) {
            val buf = allocArray<ByteVar>(bufSize)
            val n = llama_chat_apply_template(
                chatTemplate,
                chatMessages,
                messages.size.toULong(),
                addAssistant,
                buf,
                bufSize
            )
            if (n > bufSize) {
                bufSize = n + 1
                continue
            }
            check(n >= 0) { "llama_chat_apply_template failed" }
            return buf.toKString().substring(0, n)
        }
    }
}

@OptIn(ExperimentalForeignApi::class)
internal fun tokenToString(vocab: CPointer<llama_vocab>, token: Int): String {
    val buf = ByteArray(128)
    val n = buf.usePinned { pinned ->
        llama_token_to_piece(vocab, token, pinned.addressOf(0), 128, 0, true)
    }
    if (n <= 0) return ""
    return buf.decodeToString(0, n)
}

@OptIn(ExperimentalForeignApi::class)
internal fun decodeAndSample(
    ctx: CPointer<llama_context>,
    vocab: CPointer<llama_vocab>,
    sampler: CPointer<llama_sampler>,
    promptTokens: IntArray,
    config: GenerationConfig,
): GenerationResult {
    val textBuf = StringBuilder()
    val thinkBuf = if (config.thinking) StringBuilder() else null
    var inThinking = false

    val channelOpen = "<|channel>"
    val channelClose = "<channel|>"

    // Decode prompt
    promptTokens.usePinned { pinned ->
        val batch = llama_batch_get_one(pinned.addressOf(0), promptTokens.size)
        val rc = llama_decode(ctx, batch)
        check(rc == 0) { "llama_decode failed on prompt: $rc" }
    }

    val maxTokens = if (config.maxTokens > 0) config.maxTokens else Config.DEFAULT_MAX_TOKENS
    val maxGenerate = maxTokens - promptTokens.size

    for (i in 0 until maxGenerate) {
        val token = llama_sampler_sample(sampler, ctx, -1)
        llama_sampler_accept(sampler, token)

        if (llama_vocab_is_eog(vocab, token)) break

        val piece = tokenToString(vocab, token)

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

        // Feed token back for next iteration
        memScoped {
            val tokenArr = allocArray<IntVar>(1)
            tokenArr[0] = token
            val batch = llama_batch_get_one(tokenArr, 1)
            val rc = llama_decode(ctx, batch)
            if (rc != 0) break
        }
    }

    return GenerationResult(
        text = textBuf.toString(),
        thinking = thinkBuf?.toString()?.takeIf { it.isNotEmpty() },
    )
}
