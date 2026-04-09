package com.llama4j.model

import com.llama4j.tokenizer.GemmaTokenizer

class GemmaChatFormat(private val tokenizer: GemmaTokenizer) {
  private val beginOfSentence: Int
  private val startOfTurn: Int
  private val endOfTurn: Int
  private val endOfSentence: Int
  private val fimSuffix: Int
  private val fimPrefix: Int
  private val fimMiddle: Int
  private val fileSeparator: Int

  init {
    val specialTokens: Map<String, Int> = this.tokenizer.specialTokens
    this.beginOfSentence = specialTokens["<bos>"]!!
    this.startOfTurn = specialTokens["<|turn>"]!!
    this.endOfTurn = specialTokens["<turn|>"]!!
    this.endOfSentence = specialTokens["<eos>"]!!

    this.fimSuffix = specialTokens.getOrDefault("<|fim_suffix|>", -1)
    this.fimPrefix = specialTokens.getOrDefault("<|fim_prefix|>", -1)
    this.fimMiddle = specialTokens.getOrDefault("<|fim_middle|>", -1)
    this.fileSeparator = specialTokens.getOrDefault("<|file_separator|>", -1)
  }

  val stopTokens: MutableSet<Int>
    get() {
      val tokens: MutableSet<Int> = HashSet()
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
    val tokens: MutableList<Int> = ArrayList()
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
    val tokens: MutableList<Int> = ArrayList(encodeHeader(Message(Role.SYSTEM, "")))
    val thinkToken = tokenizer.specialTokens["<|think|>"]
    if (thinkToken != null) {
      tokens.add(thinkToken)
    }
    if (!systemPrompt.isNullOrEmpty()) {
      tokens.addAll(tokenizer.encode(systemPrompt.trim { it <= ' ' }))
    }
    tokens.add(endOfTurn)
    tokens.addAll(tokenizer.encode("\n"))
    return tokens
  }


  fun encodeFillInTheMiddle(prefix: String, suffix: String): MutableList<Int> {
    val tokens: MutableList<Int> = ArrayList()
    tokens.add(this.fimPrefix)
    tokens.addAll(tokenizer.encode(prefix))
    tokens.add(this.fimSuffix)
    tokens.addAll(tokenizer.encode(suffix))
    tokens.add(this.fimMiddle)
    return tokens
  }
}
