package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.data.MemorySegment
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.toUnsignedInt
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.gguf.QK_K
import io.github.lbenedetto.internal.util.vectorMathEnabled

internal class Q4_KFloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloat(index: Long): Float {
    val blockIndex: Long = index / BLOCK_SIZE
    val withinBlock = (index % BLOCK_SIZE).toInt()
    val blockOffset: Long = blockIndex * TYPE_SIZE
    val d: Float = memorySegment.readFloat16(blockOffset)
    val dmin: Float = memorySegment.readFloat16(blockOffset + 2)
    val scalesOffset = blockOffset + 4
    val qsOffset = blockOffset + 16 // 4 + 12

    // Each group of 64 values uses 2 sub-blocks: low nibble (32) + high nibble (32)
    val group = withinBlock / 64 // 0..3
    val inGroup = withinBlock % 64
    val subBlock: Int
    val nibbleIndex: Int
    val isHigh: Boolean
    if (inGroup < 32) {
      subBlock = group * 2
      nibbleIndex = inGroup
      isHigh = false
    } else {
      subBlock = group * 2 + 1
      nibbleIndex = inGroup - 32
      isHigh = true
    }

    val sc: Int = getScaleMinK4(subBlock, memorySegment, scalesOffset, false)
    val m: Int = getScaleMinK4(subBlock, memorySegment, scalesOffset, true)

    val qsByte: Byte = memorySegment.readByte(qsOffset + group * 32 + nibbleIndex)
    val quant = if (isHigh) ((qsByte.toUnsignedInt() shr 4) and 0xF) else (qsByte.toUnsignedInt() and 0xF)

    return d * sc * quant - dmin * m
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (vectorMathEnabled()) {
      Q4_KFloatTensorMath.vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }

  companion object {
    const val BLOCK_SIZE: Int = QK_K
    val TYPE_SIZE: Int = GGMLType.Q4_K.typeSize

    // Decode scale or min for sub-block j (0..7) from the 12-byte scales array
    fun getScaleMinK4(j: Int, mem: MemorySegment, scalesOffset: Long, isMin: Boolean): Int {
      if (j < 4) {
        val idx = if (isMin) j + 4 else j
        return mem.readByte(scalesOffset + idx).toUnsignedInt() and 63
      } else {
        val lowIdx = j + 4
        val highIdx = if (isMin) j else j - 4
        val low = if (isMin)
          (mem.readByte(scalesOffset + lowIdx).toUnsignedInt() shr 4)
        else
          (mem.readByte(scalesOffset + lowIdx).toUnsignedInt() and 0xF)
        val high = (mem.readByte(scalesOffset + highIdx).toUnsignedInt() shr 6) and 0x3
        return low or (high shl 4)
      }
    }

  }
}
