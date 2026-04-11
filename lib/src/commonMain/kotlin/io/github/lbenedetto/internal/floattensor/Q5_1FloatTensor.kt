package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.data.MemorySegment
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.toUnsignedInt
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.util.assert
import io.github.lbenedetto.internal.util.vectorMathEnabled

internal class Q5_1FloatTensor(
  override val size: Long,
  internal val memorySegment: MemorySegment
) : FloatTensor {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloat(index: Long): Float {
    assert(index in 0..<size)
    val blockIndex = index / GGMLType.Q5_1.blockSize
    val inBlockIndex = (index % GGMLType.Q5_1.blockSize).toInt()
    val blockOffset = blockIndex * GGMLType.Q5_1.typeSize

    val d: Float = memorySegment.readFloat16(blockOffset)
    val m: Float = memorySegment.readFloat16(blockOffset + Float16.BYTES)
    val qh: Int = readInt32LE(memorySegment, blockOffset + 2L * Float16.BYTES)

    val j: Int
    val nibble: Int
    val xh: Int
    if (inBlockIndex < GGMLType.Q5_1.blockSize / 2) {
      j = inBlockIndex
      nibble = memorySegment.readByte(
        blockOffset + 2L * Float16.BYTES + Int.SIZE_BYTES + j
      ).toUnsignedInt() and 0x0F
      xh = ((qh shr j) shl 4) and 0x10
    } else {
      j = inBlockIndex - GGMLType.Q5_1.blockSize / 2
      nibble = (memorySegment.readByte(
        blockOffset + 2L * Float16.BYTES + Int.SIZE_BYTES + j
      ).toUnsignedInt() ushr 4) and 0x0F
      xh = (qh shr (j + 12)) and 0x10
    }

    val q = nibble or xh
    return q * d + m
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor) {
      if (vectorMathEnabled()) {
        return Q5_1FloatTensorMath.vectorDot(this, thisOffset, that, thatOffset, size)
      }
      return scalarDot(this, thisOffset, that, thatOffset, size)
    }
    return scalarDot(this, thisOffset, that, thatOffset, size)
  }

  companion object {

    internal fun scalarDot(
      thiz: Q5_1FloatTensor,
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

    internal fun readInt32LE(memorySegment: MemorySegment, offset: Long): Int {
      val b0 = memorySegment.readByte(offset).toUnsignedInt()
      val b1 = memorySegment.readByte(offset + 1).toUnsignedInt()
      val b2 = memorySegment.readByte(offset + 2).toUnsignedInt()
      val b3 = memorySegment.readByte(offset + 3).toUnsignedInt()
      return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }
  }
}
