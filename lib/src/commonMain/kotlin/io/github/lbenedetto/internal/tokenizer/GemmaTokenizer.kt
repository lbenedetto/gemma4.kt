package io.github.lbenedetto.internal.tokenizer

import io.github.lbenedetto.internal.util.*

internal class GemmaTokenizer(
    private val vocabulary: Vocabulary,
    tokenType: IntArray
) {
    val specialTokens: Map<String, Int>
    private val tokenType: IntArray = tokenType.copyOf()
    private val byte0: Int

    fun isSpecialToken(tokenIndex: Int): Boolean {
        return tokenType[tokenIndex] != 1
    }

    init {
        val endOfTurn = vocabulary.getIndex("<turn|>")!!
        for (i in 0..endOfTurn) {
            if (this.tokenType[i] == 1) {
                this.tokenType[i] = 6
            }
        }
        this.byte0 = vocabulary.getIndex("<0x00>")!!
        this.specialTokens = buildSpecialTokens(this.tokenType)
            .associateBy { vocabulary[it] }
    }

    fun encode(text: String): List<Int> {
        return encodeImpl(text.replace(' ', '\u2581'))
    }

    private fun encodeImpl(text: String): List<Int> {
        val tokens = mutableListOf<Int>()

        var i = 0
        while (i < text.length) {
            val cpi = text.codePointAt(i)
            val singleCodepoint = cpi.codePointToString()
            val id = vocabulary.getIndex(singleCodepoint) ?: -1

            if (id != -1) {
                tokens.add(id)
            } else {
                for (b in singleCodepoint.encodeToByteArray()) {
                    tokens.add(b.toUByte().toInt() + byte0)
                }
            }
            i += cpi.charCount()
        }

        while (true) {
            var bestScore = -1e10f
            var bestId = -1
            var bestIdx = -1

            for (i in 0..<tokens.size - 1) {
                val strBuffer = vocabulary[tokens[i]] + vocabulary[tokens[i + 1]]
                val id = vocabulary.getIndex(strBuffer) ?: -1
                if (id != -1 && vocabulary.getScore(id) > bestScore) {
                    bestScore = vocabulary.getScore(id)
                    bestId = id
                    bestIdx = i
                }
            }

            if (bestIdx == -1) {
                break
            }

            tokens[bestIdx] = bestId
            tokens.removeAt(bestIdx + 1)
        }

        return tokens
    }

    fun decode(tokens: List<Int>): String {
        val sb = StringBuilder()
        for (token in tokens) {
            var tokenString = vocabulary[token]
            if (isSpecialToken(token)) {
                val prefix = "<0x"
                val suffix = ">"
                if (tokenString.length == 6 && tokenString.startsWith(prefix) && tokenString.endsWith(suffix)) {
                    val code = tokenString.substring(prefix.length, tokenString.length - suffix.length)
                    val cp = code.toInt(16)
                    tokenString = cp.toChar().toString()
                }
            } else {
                tokenString = tokenString.replace('\u2581', ' ')
            }
            sb.append(tokenString)
        }
        return sb.toString()
    }

    companion object {
        private fun buildSpecialTokens(tokenType: IntArray): List<Int> {
            return tokenType.indices.filter { tokenType[it] != 1 }
        }

        fun replaceControlCharacters(codePoints: IntArray): String {
            val chars = StringBuilder()
            for (cp in codePoints) {
                if (cp.isControlCodePoint() && cp != '\n'.code) {
                    chars.append("\\u").append(cp.toString(16).padStart(4, '0'))
                } else {
                    chars.append(cp.codePointToString())
                }
            }
            return chars.toString()
        }

        fun replaceControlCharacters(str: String): String {
            return replaceControlCharacters(str.toCodePoints())
        }
    }
}
