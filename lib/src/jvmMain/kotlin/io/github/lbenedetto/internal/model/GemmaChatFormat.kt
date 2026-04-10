package io.github.lbenedetto.internal.model

import io.github.lbenedetto.internal.tokenizer.GemmaTokenizer

internal class GemmaChatFormat(private val tokenizer: GemmaTokenizer) {
  private val beginOfSentence = tokenizer.specialTokens["<bos>"]!!
  private val startOfTurn = tokenizer.specialTokens["<|turn>"]!!
  private val endOfTurn = tokenizer.specialTokens["<turn|>"]!!
  private val endOfSentence = tokenizer.specialTokens["<eos>"]!!
  private val fimSuffix = tokenizer.specialTokens["<|fim_suffix|>"] ?: -1
  private val fimPrefix = tokenizer.specialTokens["<|fim_prefix|>"] ?: -1
  private val fimMiddle = tokenizer.specialTokens["<|fim_middle|>"] ?: -1
  private val fileSeparator = tokenizer.specialTokens["<|file_separator|>"] ?: -1

  val stopTokens: Set<Int> = buildSet {
    add(endOfSentence)
    add(endOfTurn)
    addAll(listOf(fimSuffix, fimPrefix, fimMiddle, fileSeparator).filter { it != -1 })
  }

  fun encodeHeader(message: Message): List<Int> = buildList {
    add(startOfTurn)
    addAll(tokenizer.encode(message.role.toString()))
    addAll(tokenizer.encode("\n"))
  }

  fun encodeMessage(message: Message): List<Int> = buildList {
    addAll(encodeHeader(message))
    addAll(tokenizer.encode(message.content.trim()))
    add(endOfTurn)
    addAll(tokenizer.encode("\n"))
  }

  fun encodeSystemThinkingTurn(systemPrompt: String?): List<Int> = buildList {
    // Matches Gemma4 template with enable_thinking=true:
    // <|turn>system\n<|think|>[system_content]<turn|>\n
    addAll(encodeHeader(Message(Role.SYSTEM, "")))
    tokenizer.specialTokens["<|think|>"]?.let { add(it) }
    if (!systemPrompt.isNullOrEmpty()) {
      addAll(tokenizer.encode(systemPrompt.trim()))
    }
    add(endOfTurn)
    addAll(tokenizer.encode("\n"))
  }

  fun encodeFillInTheMiddle(prefix: String, suffix: String): List<Int> {
    require(fimPrefix != -1 && fimSuffix != -1 && fimMiddle != -1) {
      "This model does not support fill-in-the-middle (FIM special tokens not found in vocabulary)"
    }
    return buildList {
      add(fimPrefix)
      addAll(tokenizer.encode(prefix))
      add(fimSuffix)
      addAll(tokenizer.encode(suffix))
      add(fimMiddle)
    }
  }
}
