package com.llama4j.tokenizer

import java.util.*
import java.util.function.Function
import java.util.stream.Collectors
import java.util.stream.IntStream

@JvmRecord
data class Vocabulary(tokens: Array<String>, scores: FloatArray, tokenToIndex: MutableMap<String, Int>) {
  constructor(vocabulary: Array<String>, scores: FloatArray) : this(
    vocabulary, scores,
    IntStream.range(0, vocabulary.size)
      .boxed()
      .collect(Collectors.toMap(Function { i: Int? -> vocabulary[i!!] }, Function { i: Int? -> i }))
  )

  fun get(tokenIndex: Int): String {
    return tokens[tokenIndex]
  }

  fun getIndex(token: String): OptionalInt {
    val value = tokenToIndex.get(token)
    return if (value != null) OptionalInt.of(value) else OptionalInt.empty()
  }

  fun size(): Int {
    return tokens.size
  }

  fun getScore(tokenIndex: Int): Float {
    return scores[tokenIndex]
  }

  val tokens: Array<String>
  val scores: FloatArray
  val tokenToIndex: MutableMap<String, Int>

  init {
    this.tokens = tokens
    this.scores = scores
    this.tokenToIndex = tokenToIndex
  }
}
