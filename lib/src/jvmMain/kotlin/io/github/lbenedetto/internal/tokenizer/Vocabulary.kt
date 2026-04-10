package io.github.lbenedetto.internal.tokenizer

internal data class Vocabulary(
  val tokens: Array<String>,
  val scores: FloatArray,
  val tokenToIndex: Map<String, Int>
) {
  constructor(vocabulary: Array<String>, scores: FloatArray) : this(
    tokens = vocabulary,
    scores = scores,
    tokenToIndex = vocabulary.withIndex().associate { (i, token) -> token to i }
  )

  operator fun get(tokenIndex: Int): String = tokens[tokenIndex]
  fun getIndex(token: String): Int? = tokenToIndex[token]
  fun size(): Int = tokens.size
  fun getScore(tokenIndex: Int): Float = scores[tokenIndex]
}
