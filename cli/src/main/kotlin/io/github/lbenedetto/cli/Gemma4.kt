package io.github.lbenedetto.cli

import io.github.lbenedetto.api.GemmaModel
import io.github.lbenedetto.api.GenerationConfig
import io.github.lbenedetto.api.GenerationResult

object Gemma4 {
  private const val ANSI_GREY = "\u001b[90m"
  private const val ANSI_RESET = "\u001b[0m"

  fun supportsAnsiColors(colorMode: String) = when (colorMode) {
    "on" -> true
    "auto" -> System.getenv("NO_COLOR") == null && System.getenv("TERM")?.lowercase() != "dumb"
    else -> false
  }

  private fun printThinking(result: GenerationResult, options: Options) {
    val thinking = result.thinking?.takeIf { it.isNotEmpty() } ?: return
    val out = if (options.thinkInline) System.out else System.err
    val grey = if (options.colors) ANSI_GREY else ""
    val reset = if (options.colors) ANSI_RESET else ""
    out.run {
      println("""
        $grey
        [Start thinking]
        $thinking
        [End thinking]
        $reset
      """.trimIndent())
    }
  }

  private fun GenerationConfig.applyOptions(options: Options) {
    temperature = options.temperature
    topP = options.topp
    seed = options.seed
    maxTokens = options.maxTokens
    systemPrompt = options.systemPrompt
    thinking = options.think
    if (options.stream) {
      onToken = { piece -> print(piece) }
      if (options.think) {
        val thoughtOut = if (options.thinkInline) System.out else System.err
        var emitted = false
        onThinkingStart = {
          emitted = false
          if (options.colors) thoughtOut.print(ANSI_GREY)
          thoughtOut.println("[Start thinking]")
        }
        onThinkingToken = { piece -> thoughtOut.print(piece); emitted = true }
        onThinkingEnd = {
          if (emitted) thoughtOut.println()
          thoughtOut.println("[End thinking]")
          if (options.colors) thoughtOut.print(ANSI_RESET)
          thoughtOut.println()
        }
      }
    }
  }

  fun runInteractive(model: GemmaModel, options: Options) {
    val chat = model.chat { applyOptions(options) }

    while (true) {
      print("> ")
      System.out.flush()
      val userText = readlnOrNull() ?: break

      when (userText) {
        "/quit", "/exit" -> break
        "/context" -> {
          println("${chat.contextUsed} out of ${options.maxTokens} context tokens used (${chat.contextRemaining} tokens remaining)")
          continue
        }
      }

      val result = chat.send(userText)
      if (!options.stream) printThinking(result, options)
      if (options.stream) println() else println(result.text)
      if (chat.contextRemaining <= 0) {
        System.err.println("Ran out of context length...")
        break
      }
    }
  }

  fun runInstructOnce(model: GemmaModel, options: Options) {
    val result = options.suffix
      ?.let { model.fillInMiddle(options.prompt!!, it) { applyOptions(options) } }
      ?: model.generate(options.prompt!!) { applyOptions(options) }
    if (!options.stream) printThinking(result, options)
    if (options.stream) println() else println(result.text)
  }

  @JvmStatic
  fun main(args: Array<String>) {
    val options = Options.parseOptions(args)
    val model = GemmaModel.load(options.modelPath, options.maxTokens)
    if (options.interactive) {
      runInteractive(model, options)
    } else {
      runInstructOnce(model, options)
    }
  }
}
