package io.github.lbenedetto.internal.util

internal fun String.codePointAt(index: Int): Int {
    val c1 = this[index]
    if (c1.isHighSurrogate() && index + 1 < length) {
        val c2 = this[index + 1]
        if (c2.isLowSurrogate()) {
            return 0x10000 + ((c1.code - 0xD800) shl 10) + (c2.code - 0xDC00)
        }
    }
    return c1.code
}

internal fun Int.codePointToString(): String {
    return if (this < 0x10000) {
        this.toChar().toString()
    } else {
        val cp = this - 0x10000
        val high = ((cp ushr 10) + 0xD800).toChar()
        val low = ((cp and 0x3FF) + 0xDC00).toChar()
        charArrayOf(high, low).concatToString()
    }
}

/** Returns the number of [Char] values needed to represent this code point (1 for BMP, 2 for supplementary). */
internal fun Int.charCount(): Int = if (this >= 0x10000) 2 else 1

/** Returns true if this code point is a Unicode control character (category Cc: U+0000–U+001F and U+007F–U+009F). */
internal fun Int.isControlCodePoint(): Boolean = this <= 0x1F || this in 0x7F..0x9F

internal fun String.toCodePoints(): IntArray {
    val result = mutableListOf<Int>()
    var i = 0
    while (i < length) {
        val cp = codePointAt(i)
        result.add(cp)
        i += cp.charCount()
    }
    return result.toIntArray()
}
