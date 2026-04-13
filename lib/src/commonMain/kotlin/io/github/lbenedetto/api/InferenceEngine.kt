package io.github.lbenedetto.api

import io.github.lbenedetto.internal.tokenizer.GemmaTokenizer

expect class InferenceEngine {
    internal val tokenizer: GemmaTokenizer
    fun createState(): InferenceState
    fun generateTokens(
        state: InferenceState,
        startPosition: Int,
        promptTokens: List<Int>,
        stopTokens: Set<Int>,
        maxTokens: Int,
        config: GenerationConfig,
        onToken: (Int) -> Unit,
    ): MutableList<Int>
}

expect class InferenceState
