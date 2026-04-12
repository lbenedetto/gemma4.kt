package io.github.lbenedetto.cli

import io.github.lbenedetto.api.Gemma
import io.github.lbenedetto.api.GemmaModel
import io.github.lbenedetto.api.GenerationConfig
import io.github.lbenedetto.api.GenerationResult
import okio.Path.Companion.toPath

object Gemma4 {
  private const val ANSI_GREY = "\u001b[90m"
  private const val ANSI_RESET = "\u001b[0m"

  fun supportsAnsiColors(colorMode: String) = when (colorMode) {
    "on" -> true
    "auto" -> getenv("NO_COLOR") == null && getenv("TERM")?.lowercase() != "dumb"
    else -> false
  }

  private fun printThinking(result: GenerationResult, options: Options) {
    val thinking = result.thinking?.takeIf { it.isNotEmpty() } ?: return
    val useStderr = !options.thinkInline
    val doPrintln: (String) -> Unit = if (useStderr) ::printlnStderr else ::println
    val grey = if (options.colors) ANSI_GREY else ""
    val reset = if (options.colors) ANSI_RESET else ""
    doPrintln("""
      $grey
      [Start thinking]
      $thinking
      [End thinking]
      $reset
    """.trimIndent())
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
        val useStderr = !options.thinkInline
        val doPrint: (String) -> Unit = if (useStderr) ::printStderr else ::print
        val doPrintln: (String) -> Unit = if (useStderr) ::printlnStderr else ::println
        var emitted = false
        onThinkingStart = {
          emitted = false
          if (options.colors) doPrint(ANSI_GREY)
          doPrintln("[Start thinking]")
        }
        onThinkingToken = { piece -> doPrint(piece); emitted = true }
        onThinkingEnd = {
          if (emitted) doPrintln("")
          doPrintln("[End thinking]")
          if (options.colors) doPrint(ANSI_RESET)
          doPrintln("")
        }
      }
    }
  }

  fun runInteractive(model: Gemma, options: Options) {
    val chat = model.chat { applyOptions(options) }

    while (true) {
      print("> ")
      flushStdout()
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
        printlnStderr("Ran out of context length...")
        break
      }
    }
  }

  fun runInstructOnce(model: Gemma, options: Options) {
    val result = options.suffix
      ?.let { model.fillInMiddle(options.prompt!!, it) { applyOptions(options) } }
      ?: model.generate(options.prompt!!) { applyOptions(options) }
    if (!options.stream) printThinking(result, options)
    if (options.stream) println() else println(result.text)
  }
}

fun main(args: Array<String>) {
  val options = Options.parseOptions(args)
  val model = GemmaModel.load(options.modelPath.toPath(), options.maxTokens)
  if (options.interactive) {
    Gemma4.runInteractive(model, options)
  } else {
    Gemma4.runInstructOnce(model, options)
  }
}
