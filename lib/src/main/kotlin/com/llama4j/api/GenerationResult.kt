package com.llama4j.api

/**
 * The result of a generation call.
 *
 * @property text The generated response text.
 * @property thinking The model's internal reasoning, or null if [GenerationConfig.thinking] was
 *   false or the model does not support it.
 */
data class GenerationResult(
    val text: String,
    val thinking: String? = null,
)
