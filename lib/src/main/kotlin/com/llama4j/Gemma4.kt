/**usr/bin/env jbang "$0" "$@" ; exit $? */ //JAVA 21+
//PREVIEW
//COMPILE_OPTIONS --add-modules=jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0
//MAIN com.llama4j.Gemma4
// Gemma 4 inference in pure Java
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects
// Related project: https://github.com/mukel/llama3.java
//
// Supports GGUF models and multiple tensor formats
// Matrix-vector kernels use Java's Vector API
// CLI modes: --chat and --instruct
//
// Run:
// jbang Gemma4.java --help
package com.llama4j

import com.llama4j.floattensor.FloatTensor
import com.llama4j.gguf.AOT
import com.llama4j.gguf.ModelLoader
import com.llama4j.model.*
import com.llama4j.sampler.CategoricalSampler
import com.llama4j.sampler.Sampler
import com.llama4j.sampler.ToppSampler
import com.llama4j.tokenizer.GemmaTokenizer
import java.io.IOException
import java.io.PrintStream
import java.util.*
import java.util.function.IntConsumer
import java.util.random.RandomGeneratorFactory

object Gemma4 {
  fun selectSampler(vocabularySize: Int, temperature: Float, topp: Float, rngSeed: Long): Sampler {
    val sampler: Sampler
    if (temperature == 0.0f) {
      sampler = Sampler.ARGMAX
    } else {
      val rng = RandomGeneratorFactory.getDefault().create(rngSeed)
      val innerSampler: Sampler?
      if (topp <= 0 || topp >= 1) {
        innerSampler = CategoricalSampler(rng)
      } else {
        innerSampler = ToppSampler(vocabularySize, topp, rng)
      }
      sampler = Sampler { logits: FloatTensor? ->
        val logitsSize = Math.toIntExact(logits!!.size())
        logits.divideInPlace(0, logitsSize, temperature)
        logits.softmaxInPlace(0, logitsSize)
        innerSampler.sampleToken(logits)
      }
    }
    return sampler
  }

  private const val ANSI_GREY = "\u001b[90m"
  private const val ANSI_RESET = "\u001b[0m"

  private fun plainStreamingPrinter(tokenizer: GemmaTokenizer): IntConsumer {
    return IntConsumer { token: Int ->
      if (!tokenizer.isSpecialToken(token)) {
        print(tokenizer.decode(listOf(token)))
      }
    }
  }

  private fun onThinkingStart(thoughtOut: PrintStream, ansi: Boolean) {
    if (ansi) {
      thoughtOut.print(ANSI_GREY)
    }
    thoughtOut.println("[Start thinking]")
  }

  private fun onThinkingEnd(thoughtOut: PrintStream, ansi: Boolean, emitted: Boolean) {
    if (emitted) {
      thoughtOut.println()
    }
    thoughtOut.println("[End thinking]")
    if (ansi) {
      thoughtOut.print(ANSI_RESET)
    }
    thoughtOut.println()
  }

  fun supportsAnsiColors(colorMode: String): Boolean {
    return when (colorMode) {
      "on" -> true
      "auto" -> {
        val noColor = System.getenv("NO_COLOR")
        if (noColor != null) {
          return false
        }
        val term = System.getenv("TERM")
        !"dumb".equals(term, ignoreCase = true)
      }

      else -> false
    }
  }

  private fun streamingPrinter(tokenizer: GemmaTokenizer, options: Options): IntConsumer {
    if (!options.stream) {
      return IntConsumer { }
    }

    val channelOpen = tokenizer.specialTokens["<|channel>"]
    val channelClose = tokenizer.specialTokens["<channel|>"]
    if (channelOpen == null || channelClose == null) {
      return plainStreamingPrinter(tokenizer)
    }

    val thinkEnabled = options.think
    val thoughtOut = if (options.thinkInline) System.out else System.err
    val ansi = options.colors
    val inChannel = booleanArrayOf(false)
    val emitted = booleanArrayOf(false)
    return IntConsumer { token: Int ->
      if (token == channelOpen) {
        if (thinkEnabled) {
          onThinkingStart(thoughtOut, ansi)
        }
        inChannel[0] = true
        emitted[0] = false
        return@IntConsumer
      }
      if (token == channelClose) {
        if (thinkEnabled) {
          onThinkingEnd(thoughtOut, ansi, emitted[0])
        }
        inChannel[0] = false
        emitted[0] = false
        return@IntConsumer
      }
      if (!tokenizer.isSpecialToken(token)) {
        val text = tokenizer.decode(listOf(token))
        if (inChannel[0]) {
          if (thinkEnabled) {
            thoughtOut.print(text)
            emitted[0] = true
          }
        } else {
          print(text)
        }
      }
    }
  }

  private fun visibleTokens(tokenizer: GemmaTokenizer, tokens: MutableList<Int>, think: Boolean): MutableList<Int> {
    return if (think) stripThoughtChannelTokens(tokenizer, tokens) else tokens
  }

  private fun stripThoughtChannelTokens(tokenizer: GemmaTokenizer, tokens: MutableList<Int>): MutableList<Int> {
    val channelOpen = tokenizer.specialTokens.get("<|channel>")
    val channelClose = tokenizer.specialTokens.get("<channel|>")
    if (channelOpen == null || channelClose == null || tokens.isEmpty()) {
      return tokens
    }
    val out: MutableList<Int> = ArrayList(tokens.size)
    var inChannel = false
    for (tok in tokens) {
      if (tok == channelOpen) {
        inChannel = true
        continue
      }
      if (tok == channelClose) {
        inChannel = false
        continue
      }
      if (!inChannel) {
        out.add(tok)
      }
    }
    return out
  }

  fun runInteractive(model: Llama, sampler: Sampler, options: Options) {
    var state: LlamaState? = null
    val chatFormat = GemmaChatFormat(model.tokenizer)
    val conversationTokens: MutableList<Int> = ArrayList()
    if (options.think) {
      conversationTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt))
    } else if (options.systemPrompt != null) {
      conversationTokens.addAll(chatFormat.encodeMessage(Message(Role.SYSTEM, options.systemPrompt)))
    }
    var startPosition = 0
    val `in` = Scanner(System.`in`)
    loop@ while (true) {
      print("> ")
      System.out.flush()
      val userText = `in`.nextLine()
      when (userText) {
        "/quit", "/exit" -> break@loop
        "/context" -> {
          System.out.printf(
            "%d out of %d context tokens used (%d tokens remaining)%n",
            conversationTokens.size,
            options.maxTokens,
            options.maxTokens - conversationTokens.size
          )
          continue
        }
      }
      if (state == null) {
        state = model.createNewState()
      }
      conversationTokens.addAll(chatFormat.encodeMessage(Message(Role.USER, userText)))
      conversationTokens.addAll(chatFormat.encodeHeader(Message(Role.MODEL, "")))

      val stopTokens = chatFormat.stopTokens
      val printer = streamingPrinter(model.tokenizer, options)
      val responseTokens: MutableList<Int> = Llama.generateTokens(
        model,
        state,
        startPosition,
        conversationTokens.subList(startPosition, conversationTokens.size),
        stopTokens,
        options.maxTokens,
        sampler,
        options.echo,
        options.colors,
        printer
      )
      var stopToken: Int? = null
      if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.last())) {
        stopToken = responseTokens.last()
        responseTokens.removeLast()
      }
      val visibleResponseTokens = visibleTokens(model.tokenizer, responseTokens, options.think)
      conversationTokens.addAll(responseTokens)
      if (stopToken != null) {
        conversationTokens.add(stopToken)
      }
      startPosition = conversationTokens.size
      if (!options.stream) {
        val responseText = model.tokenizer.decode(visibleResponseTokens)
        println(responseText)
      }
      if (stopToken == null) {
        System.err.println("Ran out of context length...")
        break
      }
    }
  }

  fun runInstructOnce(model: Llama, sampler: Sampler, options: Options) {
    val state = model.createNewState()
    val chatFormat = GemmaChatFormat(model.tokenizer)
    val promptTokens: MutableList<Int> = ArrayList()

    if (options.suffix != null) {
      promptTokens.addAll(chatFormat.encodeFillInTheMiddle(options.prompt!!, options.suffix))
    } else {
      if (options.think) {
        promptTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt))
      } else if (options.systemPrompt != null) {
        promptTokens.addAll(chatFormat.encodeMessage(Message(Role.SYSTEM, options.systemPrompt)))
      }
      promptTokens.addAll(chatFormat.encodeMessage(Message(Role.USER, options.prompt!!)))
      promptTokens.addAll(chatFormat.encodeHeader(Message(Role.MODEL, "")))
    }

    val stopTokens = chatFormat.stopTokens
    val printer = streamingPrinter(model.tokenizer, options)
    val responseTokens: MutableList<Int> = Llama.generateTokens(
      model,
      state,
      0,
      promptTokens,
      stopTokens,
      options.maxTokens,
      sampler,
      options.echo,
      options.colors,
      printer
    )
    if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.last())) {
      responseTokens.removeLast()
    }
    val visibleResponseTokens = visibleTokens(model.tokenizer, responseTokens, options.think)
    if (!options.stream) {
      val responseText = model.tokenizer.decode(visibleResponseTokens)
      println(responseText)
    }
  }

  @Throws(IOException::class)
  @JvmStatic
  fun main(args: Array<String>) {
    val options: Options = Options.parseOptions(args)
    var model = AOT.tryUsePreLoaded(options.modelPath, options.maxTokens)
    if (model == null) {
      model = ModelLoader.loadModel(options.modelPath, options.maxTokens)
    }
    val sampler = selectSampler(model.configuration.vocabularySize, options.temperature, options.topp, options.seed)
    if (options.interactive) {
      runInteractive(model, sampler, options)
    } else {
      runInstructOnce(model, sampler, options)
    }
  }
}
