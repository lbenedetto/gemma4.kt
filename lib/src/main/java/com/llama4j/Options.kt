package com.llama4j

import java.io.PrintStream
import java.nio.file.Path
import kotlin.system.exitProcess

@JvmRecord
data class Options(
  val modelPath: Path,
  val prompt: String?,
  val suffix: String?,
  val systemPrompt: String?,
  val interactive: Boolean,
  val temperature: Float,
  val topp: Float,
  val seed: Long,
  val maxTokens: Int,
  val stream: Boolean,
  val echo: Boolean,
  val think: Boolean,
  val thinkInline: Boolean,
  val colors: Boolean
) {
  companion object {
    const val DEFAULT_MAX_TOKENS: Int = 1024

    fun require(condition: Boolean, message: () -> String) {
      if (!condition) {
        println("ERROR ${message()}")
        println()
        printUsage(System.out)
        exitProcess(-1)
      }
    }

    fun parseBooleanOption(optionName: String, value: String): Boolean {
      return when (value.lowercase()) {
        "true", "on" -> true
        "false", "off" -> false
        else -> {
          require(false) { "Invalid argument for $optionName: expected true|false|on|off, got $value" }
          false
        }
      }
    }

    fun printUsage(out: PrintStream) {
      out.println("Usage:  jbang Gemma4.java [options]")
      out.println()
      out.println("Options:")
      out.println("  --model, -m <path>            required, path to .gguf file")
      out.println("  --interactive, --chat, -i     run in chat mode")
      out.println("  --instruct                    run in instruct (once) mode, default mode")
      out.println("  --prompt, -p <string>         input prompt")
      out.println("  --suffix <string>             suffix for fill-in-the-middle request")
      out.println("  --system-prompt, -sp <string> system prompt for chat/instruct mode")
      out.println("  --temperature, -temp <float>  temperature in [0,inf], default 1.0")
      out.println("  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95")
      out.println("  --seed <long>                 random seed, default System.nanoTime()")
      out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default $DEFAULT_MAX_TOKENS")
      out.println("  --stream <boolean>            print tokens during generation; accepts true|false|on|off, default true")
      out.println("  --echo <boolean>              print ALL tokens to stderr; accepts true|false|on|off, default false")
      out.println("  --color <on|off|auto>         colorize thinking output in terminal (default: auto)")
      out.println("  --think <off|on|inline>       off: disable thoughts, on: thoughts to stderr, inline: thoughts to stdout")
      out.println()
      out.println("Interactive commands:")
      out.println("  /quit, /exit                  exit the chat")
      out.println("  /context                      show context token usage")
      out.println()
      out.println("Examples:")
      out.println("  jbang Gemma4.java --model gemma-4-E2B-it-Q8_0.gguf --chat")
      out.println("  jbang Gemma4.java --model gemma-4-E2B-it-Q8_0.gguf --prompt \"Tell me a joke\"")
      out.println("  jbang Gemma4.java --model gemma-4-E2B-it-Q8_0.gguf --chat --system-prompt \"You are a helpful assistant\"")
    }

    fun parseOptions(args: Array<String>): Options {
      var prompt: String? = null
      var suffix: String? = null
      var systemPrompt: String? = null
      var temperature = 1f
      var topp = 0.95f
      var modelPath: Path? = null
      var seed = System.nanoTime()
      var maxTokens: Int = DEFAULT_MAX_TOKENS
      var interactive = false
      var stream = true
      var echo = false
      var think = false
      var thinkInline = false
      var colorMode = "auto"

      var i = 0
      while (i < args.size) {
        var optionName = args[i]
        require(optionName.startsWith("-")) { "Invalid option $optionName" }
        when (optionName) {
          "--interactive", "--chat", "-i" -> interactive = true
          "--instruct" -> interactive = false
          "--help", "-h" -> {
            printUsage(System.out)
            System.exit(0)
          }

          else -> {
            val nextArg: String?
            if (optionName.contains("=")) {
              val parts = optionName.split("=".toRegex(), limit = 2).toTypedArray()
              optionName = parts[0]
              nextArg = parts[1]
            } else {
              require(i + 1 < args.size) { "Missing argument for option $optionName" }
              nextArg = args[i + 1]
              i += 1
            }
            when (optionName) {
              "--prompt", "-p" -> prompt = nextArg
              "--suffix" -> suffix = nextArg
              "--system-prompt", "-sp" -> systemPrompt = nextArg
              "--temperature", "--temp" -> temperature = nextArg.toFloat()
              "--top-p" -> topp = nextArg.toFloat()
              "--model", "-m" -> modelPath = Path.of(nextArg)
              "--seed", "-s" -> seed = nextArg.toLong()
              "--max-tokens", "-n" -> maxTokens = nextArg.toInt()
              "--stream" -> stream = parseBooleanOption(optionName, nextArg)
              "--echo" -> echo = parseBooleanOption(optionName, nextArg)
              "--color" -> colorMode = nextArg.lowercase()
              "--think" -> {
                val thinkMode = nextArg.lowercase()
                thinkInline = mutableListOf("inline", "stdout").contains(thinkMode)
                when (thinkMode) {
                  "on", "true", "inline", "stdout" -> think = true
                  "off", "false" -> think = false
                  else -> require(false) { "Invalid argument for $optionName: expected off|on|inline (or false|true|stdout), got $nextArg" }
                }
              }

              else -> require(false) { "Unknown option: $optionName" }
            }
          }
        }
        i++
      }

      require(setOf("on", "off", "auto").contains(colorMode)) { "Invalid argument: --color must be one of on|off|auto" }
      val color = Gemma4.supportsAnsiColors(colorMode)
      require(modelPath != null) { "Missing argument: --model <path> is required" }
      require(interactive || prompt != null) { "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"" }
      require(0 <= temperature) { "Invalid argument: --temperature must be non-negative" }
      require(topp in 0.0..1.0) { "Invalid argument: --top-p must be within [0, 1]" }
      return Options(
        modelPath!!,
        prompt,
        suffix,
        systemPrompt,
        interactive,
        temperature,
        topp,
        seed,
        maxTokens,
        stream,
        echo,
        think,
        thinkInline,
        color
      )
    }
  }
}
