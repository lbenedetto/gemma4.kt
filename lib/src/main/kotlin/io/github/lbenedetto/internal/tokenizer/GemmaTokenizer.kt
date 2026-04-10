package io.github.lbenedetto.internal.tokenizer

import java.nio.charset.StandardCharsets
import java.util.*

internal class GemmaTokenizer(
    private val vocabulary: Vocabulary,
    tokenType: IntArray
) {
    val specialTokens: Map<String, Int>
    private val tokenType: IntArray = tokenType.clone()
    private val byte0: Int

    fun isSpecialToken(tokenIndex: Int): Boolean {
        return tokenType[tokenIndex] != 1
    }

    init {
        val endOfTurn = vocabulary.getIndex("<turn|>").orElseThrow()
        for (i in 0..endOfTurn) {
            if (this.tokenType[i] == 1) {
                this.tokenType[i] = 6
            }
        }
        this.byte0 = vocabulary.getIndex("<0x00>").orElseThrow()
        this.specialTokens = buildSpecialTokens(this.tokenType)
            .associateBy { vocabulary[it] }
    }

    fun encode(text: String): List<Int> {
        return encodeImpl(text.replace(' ', '\u2581'))
    }

    private fun encodeImpl(text: String): List<Int> {
        val tokens = mutableListOf<Int>()

        var i = 0
        var cpi: Int
        while (i < text.length) {
            cpi = text.codePointAt(i)

            val singleCodepoint = Character.toString(cpi)
            val id = vocabulary.getIndex(singleCodepoint).orElse(-1)

            if (id != -1) {
                tokens.add(id)
            } else {
                for (b in singleCodepoint.toByteArray(StandardCharsets.UTF_8)) {
                    tokens.add(b.toUByte().toInt() + byte0)
                }
            }
            i += Character.charCount(cpi)
        }

        while (true) {
            var bestScore = -1e10f
            var bestId = -1
            var bestIdx = -1

            for (i in 0..<tokens.size - 1) {
                val strBuffer = vocabulary[tokens[i]] + vocabulary[tokens[i + 1]]
                val id = vocabulary.getIndex(strBuffer).orElse(-1)
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
                    tokenString = Character.toString(cp)
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
                if (Character.getType(cp) == Character.CONTROL.toInt() && cp != '\n'.code) {
                    chars.append("\\u").append(HexFormat.of().toHexDigits(cp.toLong(), 4))
                } else {
                    chars.appendCodePoint(cp)
                }
            }
            return chars.toString()
        }

        @JvmStatic
        fun replaceControlCharacters(str: String): String {
            return replaceControlCharacters(str.codePoints().toArray())
        }
    }
}
