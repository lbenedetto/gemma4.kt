package io.github.lbenedetto.api

actual class ChatSession {
  actual val contextUsed: Int
    get() = TODO("Not yet implemented")
  actual val contextRemaining: Int
    get() = TODO("Not yet implemented")

  actual fun send(message: String): GenerationResult {
    TODO("Not yet implemented")
  }

  actual fun reset() {
  }
}
