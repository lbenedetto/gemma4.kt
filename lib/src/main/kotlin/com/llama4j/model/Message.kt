package com.llama4j.model

@JvmRecord
data class Message(val role: Role, val content: String)
