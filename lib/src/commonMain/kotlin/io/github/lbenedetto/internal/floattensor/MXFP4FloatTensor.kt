package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.toUnsignedInt
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.gguf.QK_MXFP4
import io.github.lbenedetto.internal.util.MemorySegment
import io.github.lbenedetto.internal.util.assert
import io.github.lbenedetto.internal.util.vectorMathEnabled

internal class MXFP4FloatTensor(
  override val size: Long,
  internal val memorySegment: MemorySegment
) : FloatTensor {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloat(index: Long): Float {
    assert(index in 0..<size)
    val blockIndex: Long = index / QK_MXFP4
    val inBlockIndex = (index % QK_MXFP4).toInt()
    val blockOffset = blockIndex * GGMLType.MXFP4.typeSize

    val e8m0 = memorySegment.readByte(blockOffset).toUnsignedInt()
    val d: Float = e8m0ToFp32Half(e8m0)

    val qsOffset = blockOffset + Byte.SIZE_BYTES + (inBlockIndex and 0x0F)
    val packed = memorySegment.readByte(qsOffset).toUnsignedInt()
    val nibble = if (inBlockIndex < (QK_MXFP4 / 2)) (packed and 0x0F) else ((packed ushr 4) and 0x0F)

    return MXFP4_VALUES[nibble] * d
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor) {
      if (vectorMathEnabled()) {
        return MXFP4FloatTensorMath.vectorDot(this, thisOffset, that, thatOffset, size)
      }
      return scalarDot(this, thisOffset, that, thatOffset, size)
    }
    return scalarDot(this, thisOffset, that, thatOffset, size)
  }

  companion object {
    private val MXFP4_VALUES = intArrayOf(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

    private fun scalarDot(
      thiz: MXFP4FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      var result = 0f
      for (i in 0..<size) {
        result += thiz.getFloat((thisOffset + i).toLong()) * that.values[thatOffset + i]
      }
      return result
    }

    internal fun e8m0ToFp32Half(x: Int): Float {
      val bits: Int = if (x < 2) {
        0x00200000 shl x
      } else {
        (x - 1) shl 23
      }
      return Float.fromBits(bits)
    }
  }
}
