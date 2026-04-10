package com.llama4j.internal.model

@JvmRecord
data class Message(val role: Role, val content: String)
