package com.llama4j

import com.llama4j.api.GemmaModel
import com.llama4j.api.GenerationConfig
import com.llama4j.api.GenerationResult
import java.io.IOException
import java.io.PrintStream
import java.util.*

object Gemma4 {
  private const val ANSI_GREY = "\u001b[90m"
  private const val ANSI_RESET = "\u001b[0m"

  fun supportsAnsiColors(colorMode: String): Boolean {
    return when (colorMode) {
      "on" -> true
      "auto" -> {
        val noColor = System.getenv("NO_COLOR")
        if (noColor != null) return false
        val term = System.getenv("TERM")
        !"dumb".equals(term, ignoreCase = true)
      }
      else -> false
    }
  }

  private fun printThinking(result: GenerationResult, options: Options) {
    val thinking = result.thinking?.takeIf { it.isNotEmpty() } ?: return
    val out: PrintStream = if (options.thinkInline) System.out else System.err
    if (options.colors) out.print(ANSI_GREY)
    out.println("[Start thinking]")
    out.print(thinking)
    out.println()
    out.println("[End thinking]")
    if (options.colors) out.print(ANSI_RESET)
    out.println()
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
        val thoughtOut: PrintStream = if (options.thinkInline) System.out else System.err
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
    val scanner = Scanner(System.`in`)

    loop@ while (true) {
      print("> ")
      System.out.flush()
      val userText = scanner.nextLine()
      when (userText) {
        "/quit", "/exit" -> break@loop
        "/context" -> {
          System.out.printf(
            "%d out of %d context tokens used (%d tokens remaining)%n",
            chat.contextUsed,
            options.maxTokens,
            chat.contextRemaining
          )
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
    val result = if (options.suffix != null) {
      model.fillInMiddle(options.prompt!!, options.suffix) { applyOptions(options) }
    } else {
      model.generate(options.prompt!!) { applyOptions(options) }
    }
    if (!options.stream) printThinking(result, options)
    if (options.stream) println() else println(result.text)
  }

  @Throws(IOException::class)
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
