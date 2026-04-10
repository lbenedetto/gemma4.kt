package io.github.lbenedetto.internal.gguf

import io.github.lbenedetto.internal.floattensor.Float16

const val QK_K: Int = 256
const val QK_MXFP4: Int = 32

private fun isPowerOf2(n: Int): Boolean {
  return n > 0 && (n and (n - 1)) == 0
}

enum class GGMLType(
  val typeSize: Int,
  val blockSize: Int = 1
) {
  /* 0 */ F32(Float.SIZE_BYTES),
  /* 1 */ F16(Float16.BYTES),
  /* 2 */ Q4_0(Float16.BYTES + 16 * Byte.SIZE_BYTES, 32),
  /* 3 */ Q4_1(2 * Float16.BYTES + 16 * Byte.SIZE_BYTES, 32),
  /* 4 - removed */ UNSUPPORTED_Q4_2(Int.MAX_VALUE),
  /* 5 - removed */ UNSUPPORTED_Q4_3(Int.MAX_VALUE),
  /* 6 */ Q5_0(Int.MAX_VALUE),
  /* 7 */ Q5_1(2 * Float16.BYTES + Int.SIZE_BYTES + 16 * Byte.SIZE_BYTES, 32),
  /* 8 */ Q8_0(Float16.BYTES + 32 * Byte.SIZE_BYTES, 32),
  /* 9 */ Q8_1(32 * Byte.SIZE_BYTES + 2 * Float.SIZE_BYTES, 32),
  /* 10 */ Q2_K(Int.MAX_VALUE),
  /* 11 */ Q3_K(Int.MAX_VALUE),
  /* 12 */ Q4_K(2 * Float16.BYTES + ((QK_K / 16) / 8 * 6) + QK_K / 2, QK_K),
  /* 13 */ Q5_K(2 * Float16.BYTES + ((QK_K / 16) / 8 * 6) + QK_K / 8 + QK_K / 2, QK_K),
  /* 14 */ Q6_K(QK_K / 2 + QK_K / 4 + QK_K / 16 + Float16.BYTES, QK_K),
  /* 15 */ Q8_K(Int.MAX_VALUE),
  /* 16 */ IQ2_XXS(Int.MAX_VALUE),
  /* 17 */ IQ2_XS(Int.MAX_VALUE),
  /* 18 */ IQ3_XXS(Int.MAX_VALUE),
  /* 19 */ IQ1_S(Int.MAX_VALUE),
  /* 20 */ IQ4_NL(Int.MAX_VALUE),
  /* 21 */ IQ3_S(Int.MAX_VALUE),
  /* 22 */ IQ2_S(Int.MAX_VALUE),
  /* 23 */ IQ4_XS(Int.MAX_VALUE),
  /* 24 */ I8(Byte.SIZE_BYTES),
  /* 25 */ I16(Short.SIZE_BYTES),
  /* 26 */ I32(Int.SIZE_BYTES),
  /* 27 */ I64(Long.SIZE_BYTES),
  /* 28 */ F64(Double.SIZE_BYTES),
  /* 29 */ IQ1_M(Int.MAX_VALUE),
  /* 30 */ BF16(Float16.BYTES),
  /* 31 - removed from gguf files */ UNSUPPORTED_Q4_0_4_4(Int.MAX_VALUE),
  /* 32 */ UNSUPPORTED_Q4_0_4_8(Int.MAX_VALUE),
  /* 33 */ UNSUPPORTED_Q4_0_8_8(Int.MAX_VALUE),
  /* 34 */ TQ1_0(Int.MAX_VALUE),
  /* 35 */ TQ2_0(Int.MAX_VALUE),
  /* 36 */ UNSUPPORTED_IQ4_NL_4_4(Int.MAX_VALUE),
  /* 37 */ UNSUPPORTED_IQ4_NL_4_8(Int.MAX_VALUE),
  /* 38 */ UNSUPPORTED_IQ4_NL_8_8(Int.MAX_VALUE),
  /* 39 */ MXFP4(Byte.SIZE_BYTES + QK_MXFP4 / 2, QK_MXFP4),
  /* 40 */ NVFP4(Int.MAX_VALUE);

  fun byteSizeFor(numberOfElements: Long): Long {
    val t = numberOfElements * this.typeSize.toLong()
    assert(t % this.blockSize == 0L)
    return t / this.blockSize
  }

  init {
    assert(blockSize > 0)
    assert(typeSize > 0)
    assert(isPowerOf2(blockSize))
  }

  companion object {
    private val VALUES = entries.toTypedArray()

    fun fromId(id: Int): GGMLType {
      if (0 <= id && id < VALUES.size) {
        return VALUES[id]
      }
      throw UnsupportedOperationException("Unsupported GGML tensor type id: $id")
    }
  }
}
