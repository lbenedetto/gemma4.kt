package com.llama4j.model

import com.llama4j.tokenizer.GemmaTokenizer
import java.util.*

class GemmaChatFormat(tokenizer: GemmaTokenizer) {
  protected val tokenizer: GemmaTokenizer
  protected val beginOfSentence: Int
  protected val startOfTurn: Int
  protected val endOfTurn: Int
  protected val endOfSentence: Int
  protected val fimSuffix: Int
  protected val fimPrefix: Int
  protected val fimMiddle: Int
  protected val fileSeparator: Int

  init {
    this.tokenizer = tokenizer
    val specialTokens: MutableMap<String, Int> = this.tokenizer.specialTokens
    this.beginOfSentence = Objects.requireNonNull<Int>(specialTokens.get("<bos>"))
    this.startOfTurn = Objects.requireNonNull<Int>(specialTokens.get("<|turn>"))
    this.endOfTurn = Objects.requireNonNull<Int>(specialTokens.get("<turn|>"))
    this.endOfSentence = Objects.requireNonNull<Int>(specialTokens.get("<eos>"))

    this.fimSuffix = specialTokens.getOrDefault("<|fim_suffix|>", -1)
    this.fimPrefix = specialTokens.getOrDefault("<|fim_prefix|>", -1)
    this.fimMiddle = specialTokens.getOrDefault("<|fim_middle|>", -1)
    this.fileSeparator = specialTokens.getOrDefault("<|file_separator|>", -1)
  }

  val stopTokens: MutableSet<Int>
    get() {
      val tokens: MutableSet<Int> = HashSet<Int>()
      tokens.add(endOfSentence)
      tokens.add(endOfTurn)
      if (fimSuffix != -1) {
        tokens.add(fimSuffix)
      }
      if (fimPrefix != -1) {
        tokens.add(fimPrefix)
      }
      if (fimMiddle != -1) {
        tokens.add(fimMiddle)
      }
      if (fileSeparator != -1) {
        tokens.add(fileSeparator)
      }
      return tokens
    }

  fun encodeHeader(message: Message): MutableList<Int> {
    val tokens: MutableList<Int> = ArrayList<Int>()
    tokens.add(startOfTurn)
    tokens.addAll(tokenizer.encode(message.role.toString()))
    tokens.addAll(this.tokenizer.encode("\n"))
    return tokens
  }

  fun encodeMessage(message: Message): MutableList<Int> {
    val tokens = this.encodeHeader(message)
    tokens.addAll(this.tokenizer.encode(message.content.trim()))
    tokens.add(endOfTurn)
    tokens.addAll(this.tokenizer.encode("\n"))
    return tokens
  }

  fun encodeSystemThinkingTurn(systemPrompt: String?): MutableList<Int> {
    // Matches Gemma4 template with enable_thinking=true:
    // <|turn>system\n<|think|>[system_content]<turn|>\n
    val tokens: MutableList<Int> = ArrayList<Int>(encodeHeader(Message(Role.Companion.SYSTEM, "")))
    val thinkToken = tokenizer.specialTokens.get("<|think|>")
    if (thinkToken != null) {
      tokens.add(thinkToken)
    }
    if (systemPrompt != null && !systemPrompt.isEmpty()) {
      tokens.addAll(tokenizer.encode(systemPrompt.trim { it <= ' ' }))
    }
    tokens.add(endOfTurn)
    tokens.addAll(tokenizer.encode("\n"))
    return tokens
  }


  fun encodeFillInTheMiddle(prefix: String, suffix: String): MutableList<Int> {
    val tokens: MutableList<Int> = ArrayList<Int>()
    tokens.add(this.fimPrefix)
    tokens.addAll(tokenizer.encode(prefix))
    tokens.add(this.fimSuffix)
    tokens.addAll(tokenizer.encode(suffix))
    tokens.add(this.fimMiddle)
    return tokens
  }
}
