package io.github.lbenedetto.internal.util

import io.kotest.core.spec.style.BehaviorSpec
import io.kotest.matchers.shouldBe
import io.github.lbenedetto.internal.util.codePointAt as codePointAtImpl

class CodePointsSpec : BehaviorSpec({

    // Covers: ASCII, extended Latin, CJK (all BMP), and supplementary code points (require surrogate pairs)
    val bmpCodePoints = listOf(
        0x41,   // 'A'
        0x7A,   // 'z'
        0x00,   // NUL
        0xE9,   // 'é'
        0x2581, // '▁' (used by the tokenizer itself)
        0x4E2D, // '中'
        0xFFFF, // last BMP code point
    )
    val supplementaryCodePoints = listOf(
        0x1F600, // 😀
        0x1F9E0, // 🧠
        0x10000, // first supplementary code point
        0x10FFFF, // last valid Unicode code point
    )
    val allCodePoints = bmpCodePoints + supplementaryCodePoints

    // Strings that exercise surrogate pair boundaries
    val testStrings = listOf(
        "hello",
        "café",
        "中文",
        "▁token",
        "hi😀",
        "a😀b🧠c",
        "\u0000\u001F\n",
        "\uFFFF",
        "\uD83D\uDE00",   // 😀 as raw surrogate pair
    )

    // String.codePointAt is a JVM member function, so `str.codePointAt(i)` would call the JVM
    // implementation regardless of our import. The alias forces resolution to our extension.
    Given("codePointAt") {
        for (str in testStrings) {
            When("iterating \"${str.take(20).replace("\u0000", "\\u0000")}\"") {
                var i = 0
                while (i < str.length) {
                    val index = i
                    val jvmResult = str.codePointAt(index)
                    Then("index $index matches JVM") {
                        str.codePointAtImpl(index) shouldBe jvmResult
                    }
                    i += Character.charCount(jvmResult)
                }
            }
        }
    }

    // Int has no codePointToString member, so no shadowing — no alias needed
    Given("codePointToString") {
        for (cp in allCodePoints) {
            When("code point is U+${cp.toString(16).uppercase().padStart(4, '0')}") {
                Then("matches Character.toString(cp)") {
                    cp.codePointToString() shouldBe Character.toString(cp)
                }
            }
        }
    }

    Given("charCount") {
        for (cp in allCodePoints) {
            When("code point is U+${cp.toString(16).uppercase().padStart(4, '0')}") {
                Then("matches Character.charCount(cp)") {
                    cp.charCount() shouldBe Character.charCount(cp)
                }
            }
        }
    }

    Given("isControlCodePoint") {
        val controlCodePoints = listOf(0x00, 0x01, 0x1F, 0x7F, 0x80, 0x9F)
        val nonControlCodePoints = listOf(0x20, 0x41, 0x0A, 0xA0, 0x2581, 0x1F600)
        for (cp in controlCodePoints) {
            When("code point U+${cp.toString(16).uppercase().padStart(4, '0')} is a control character") {
                Then("matches Character.getType(cp) == Character.CONTROL") {
                    cp.isControlCodePoint() shouldBe (Character.getType(cp) == Character.CONTROL.toInt())
                }
            }
        }
        for (cp in nonControlCodePoints) {
            When("code point U+${cp.toString(16).uppercase().padStart(4, '0')} is not a control character") {
                Then("matches Character.getType(cp) == Character.CONTROL") {
                    cp.isControlCodePoint() shouldBe (Character.getType(cp) == Character.CONTROL.toInt())
                }
            }
        }
    }

    // String has no toCodePoints member, so no shadowing — no alias needed
    Given("toCodePoints") {
        for (str in testStrings) {
            When("string is \"${str.take(20).replace("\u0000", "\\u0000")}\"") {
                Then("matches str.codePoints().toArray()") {
                    str.toCodePoints() shouldBe str.codePoints().toArray()
                }
            }
        }
    }
})
