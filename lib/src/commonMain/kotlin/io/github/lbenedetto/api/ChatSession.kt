package io.github.lbenedetto.api

expect class ChatSession {
  val contextUsed: Int
  val contextRemaining: Int
  fun send(message: String): GenerationResult
  fun reset()
}
