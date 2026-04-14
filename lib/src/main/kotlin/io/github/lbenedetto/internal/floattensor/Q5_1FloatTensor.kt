package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.gguf.GGMLType
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.foreign.MemorySegment
import kotlin.math.min

internal class Q5_1FloatTensor(
  override val size: Long,
  private val memorySegment: MemorySegment
) : FloatTensor() {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector? {
    throw UnsupportedOperationException("getFloatVector")
  }

  override fun getFloat(index: Long): Float {
    assert(index in 0..<size)
    val blockIndex = index / GGMLType.Q5_1.blockSize
    val inBlockIndex = (index % GGMLType.Q5_1.blockSize).toInt()
    val blockOffset = blockIndex * GGMLType.Q5_1.typeSize

    val d: Float = readFloat16(memorySegment, blockOffset)
    val m: Float = readFloat16(memorySegment, blockOffset + Float16.BYTES)
    val qh: Int = readInt32LE(memorySegment, blockOffset + 2L * Float16.BYTES)

    val j: Int
    val nibble: Int
    val xh: Int
    if (inBlockIndex < GGMLType.Q5_1.blockSize / 2) {
      j = inBlockIndex
      nibble = readByte(
        memorySegment,
        blockOffset + 2L * Float16.BYTES + Integer.BYTES + j
      ).toUnsignedInt() and 0x0F
      xh = ((qh shr j) shl 4) and 0x10
    } else {
      j = inBlockIndex - GGMLType.Q5_1.blockSize / 2
      nibble = (readByte(
        memorySegment,
        blockOffset + 2L * Float16.BYTES + Integer.BYTES + j
      ).toUnsignedInt() ushr 4) and 0x0F
      xh = (qh shr (j + 12)) and 0x10
    }

    val q = nibble or xh
    return q * d + m
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor) {
      if (USE_VECTOR_API) {
        return vectorDot(this, thisOffset, that, thatOffset, size)
      }
      return scalarDot(this, thisOffset, that, thatOffset, size)
    }
    return scalarDot(this, thisOffset, that, thatOffset, size)
  }

  companion object {
    private fun vectorDot(
      thiz: Q5_1FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      assert(Integer.bitCount(GGMLType.Q5_1.blockSize) == 1) { "power of 2" }
      var j = 0
      var result = 0f

      val alignmentBound = min(size, -thisOffset and (GGMLType.Q5_1.blockSize - 1))
      if (alignmentBound > 0) {
        result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
        j = alignmentBound
      }

      val decoded = FloatArray(GGMLType.Q5_1.blockSize)
      val upperBound = j + (size - j) / GGMLType.Q5_1.blockSize * GGMLType.Q5_1.blockSize
      val vecUpper = F_SPECIES!!
        .loopBound(GGMLType.Q5_1.blockSize)
      while (j < upperBound) {
        assert((thisOffset + j) % GGMLType.Q5_1.blockSize == 0)
        val blockOffset = (thisOffset + j).toLong() / GGMLType.Q5_1.blockSize * GGMLType.Q5_1.typeSize
        val d: Float = readFloat16(thiz.memorySegment, blockOffset)
        val m: Float = readFloat16(thiz.memorySegment, blockOffset + Float16.BYTES)
        val qh: Int = readInt32LE(thiz.memorySegment, blockOffset + 2L * Float16.BYTES)
        val qsBase = blockOffset + 2L * Float16.BYTES + Integer.BYTES

        for (p in 0..<GGMLType.Q5_1.blockSize / 2) {
          val packed = readByte(thiz.memorySegment, qsBase + p).toUnsignedInt()
          val x0 = (packed and 0x0F) or ((((qh shr p) shl 4) and 0x10))
          val x1 = ((packed ushr 4) and 0x0F) or ((qh shr (p + 12)) and 0x10)
          decoded[p] = x0 * d + m
          decoded[p + GGMLType.Q5_1.blockSize / 2] = x1 * d + m
        }

        var acc = FloatVector.zero(F_SPECIES)
        run {
          var i = 0
          while (i < vecUpper) {
            val w = FloatVector.fromArray(F_SPECIES, decoded, i)
            val x = that.getFloatVector(F_SPECIES, thatOffset + j + i)
            acc = w.fma(x, acc)
            i += F_SPECIES.length()
          }
        }
        result += acc.reduceLanes(VectorOperators.ADD)

        for (i in vecUpper..<GGMLType.Q5_1.blockSize) {
          result += decoded[i] * that.values[thatOffset + j + i]
        }
        j += GGMLType.Q5_1.blockSize
      }

      if (j < size) {
        result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
      }
      return result
    }

    private fun scalarDot(
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

    private fun readInt32LE(memorySegment: MemorySegment, offset: Long): Int {
      val b0 = readByte(memorySegment, offset).toUnsignedInt()
      val b1 = readByte(memorySegment, offset + 1).toUnsignedInt()
      val b2 = readByte(memorySegment, offset + 2).toUnsignedInt()
      val b3 = readByte(memorySegment, offset + 3).toUnsignedInt()
      return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }
  }
}
