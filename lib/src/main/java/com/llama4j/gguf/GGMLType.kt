package com.llama4j.gguf

import com.llama4j.floattensor.Float16
import java.lang.Byte
import java.lang.Double
import java.lang.Float
import java.lang.Long
import java.lang.Short
import kotlin.Boolean
import kotlin.Int
import kotlin.UnsupportedOperationException
import kotlin.assert
import kotlin.collections.toTypedArray

enum class GGMLType @JvmOverloads constructor(typeSize: Int, blockSize: Int = 1) {
  F32(Float.BYTES),  // 0
  F16(Float16.BYTES),  // 1
  Q4_0(Float16.BYTES + 16 * Byte.BYTES, 32),  // 2
  Q4_1(2 * Float16.BYTES + 16 * Byte.BYTES, 32),  // 3
  UNSUPPORTED_Q4_2(Int.MAX_VALUE),  // 4 - removed
  UNSUPPORTED_Q4_3(Int.MAX_VALUE),  // 5 - removed
  Q5_0(Int.MAX_VALUE),  // 6
  Q5_1(2 * Float16.BYTES + Integer.BYTES + 16 * Byte.BYTES, 32),  // 7
  Q8_0(Float16.BYTES + 32 * Byte.BYTES, 32),  // 8
  Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32),  // 9
  Q2_K(Int.MAX_VALUE),  // 10
  Q3_K(Int.MAX_VALUE),  // 11
  Q4_K(
    2 * Float16.BYTES + ((GGMLType.Companion.QK_K / 16) / 8 * 6) + GGMLType.Companion.QK_K / 2,
    GGMLType.Companion.QK_K
  ),  // 12
  Q5_K(
    2 * Float16.BYTES + ((GGMLType.Companion.QK_K / 16) / 8 * 6) + GGMLType.Companion.QK_K / 8 + GGMLType.Companion.QK_K / 2,
    GGMLType.Companion.QK_K
  ),  // 13
  Q6_K(
    GGMLType.Companion.QK_K / 2 + GGMLType.Companion.QK_K / 4 + GGMLType.Companion.QK_K / 16 + Float16.BYTES,
    GGMLType.Companion.QK_K
  ),  // 14
  Q8_K(Int.MAX_VALUE),  // 15
  IQ2_XXS(Int.MAX_VALUE),  // 16
  IQ2_XS(Int.MAX_VALUE),  // 17
  IQ3_XXS(Int.MAX_VALUE),  // 18
  IQ1_S(Int.MAX_VALUE),  // 19
  IQ4_NL(Int.MAX_VALUE),  // 20
  IQ3_S(Int.MAX_VALUE),  // 21
  IQ2_S(Int.MAX_VALUE),  // 22
  IQ4_XS(Int.MAX_VALUE),  // 23
  I8(Byte.BYTES),  // 24
  I16(Short.BYTES),  // 25
  I32(Integer.BYTES),  // 26
  I64(Long.BYTES),  // 27
  F64(Double.BYTES),  // 28
  IQ1_M(Int.MAX_VALUE),  // 29
  BF16(Float16.BYTES),  // 30
  UNSUPPORTED_Q4_0_4_4(Int.MAX_VALUE),  // 31 - removed from gguf files
  UNSUPPORTED_Q4_0_4_8(Int.MAX_VALUE),  // 32
  UNSUPPORTED_Q4_0_8_8(Int.MAX_VALUE),  // 33
  TQ1_0(Int.MAX_VALUE),  // 34
  TQ2_0(Int.MAX_VALUE),  // 35
  UNSUPPORTED_IQ4_NL_4_4(Int.MAX_VALUE),  // 36
  UNSUPPORTED_IQ4_NL_4_8(Int.MAX_VALUE),  // 37
  UNSUPPORTED_IQ4_NL_8_8(Int.MAX_VALUE),  // 38
  MXFP4(Byte.BYTES + GGMLType.Companion.QK_MXFP4 / 2, GGMLType.Companion.QK_MXFP4),  // 39
  NVFP4(Int.MAX_VALUE); // 40

  val typeSize: Int

  val blockSize: Int

  fun byteSizeFor(numberOfElements: kotlin.Long): kotlin.Long {
    val t = numberOfElements * this.typeSize.toLong()
    assert(t % this.blockSize == 0L)
    return t / this.blockSize
  }

  init {
    assert(blockSize > 0)
    assert(typeSize > 0)
    assert(isPowerOf2(blockSize))
    this.typeSize = typeSize
    this.blockSize = blockSize
  }

  companion object {
    private val VALUES = entries.toTypedArray()

    fun fromId(id: Int): GGMLType {
      if (0 <= id && id < VALUES.size) {
        return VALUES[id]
      }
      throw UnsupportedOperationException("Unsupported GGML tensor type id: " + id)
    }

    const val QK_K: Int = 256
    const val QK_MXFP4: Int = 32

    private fun isPowerOf2(n: Int): Boolean {
      return n > 0 && (n and (n - 1)) == 0
    }
  }
}
