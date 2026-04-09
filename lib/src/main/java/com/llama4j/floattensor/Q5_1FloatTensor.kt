package com.llama4j.floattensor

import com.llama4j.gguf.GGMLType
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.Byte
import java.lang.foreign.MemorySegment
import java.util.*
import kotlin.Float
import kotlin.FloatArray
import kotlin.Int
import kotlin.Long
import kotlin.UnsupportedOperationException
import kotlin.assert
import kotlin.collections.minus
import kotlin.collections.plus
import kotlin.div
import kotlin.math.min
import kotlin.plus
import kotlin.run
import kotlin.sequences.minus
import kotlin.sequences.plus
import kotlin.times

internal class Q5_1FloatTensor(size: Long, memorySegment: MemorySegment) : FloatTensor() {
  private val size: Long
  private val memorySegment: MemorySegment

  init {
    this.size = size
    this.memorySegment = memorySegment
  }

  override fun size(): Long {
    return size
  }

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloatVector(species: VectorSpecies<Float>, index: Int): FloatVector? {
    throw UnsupportedOperationException("getFloatVector")
  }

  public override fun type(): GGMLType {
    return GGMLType.Q5_1
  }

  override fun getFloat(index: Long): Float {
    assert(0 <= index && index < size)
    val blockIndex = index / GGMLType.Q5_1.getBlockSize()
    val inBlockIndex = (index % GGMLType.Q5_1.getBlockSize()).toInt()
    val blockOffset = blockIndex * GGMLType.Q5_1.getTypeSize()

    val d: Float = FloatTensor.Companion.readFloat16(memorySegment, blockOffset)
    val m: Float = FloatTensor.Companion.readFloat16(memorySegment, blockOffset + Float16.BYTES)
    val qh: Int = readInt32LE(memorySegment, blockOffset + 2L * Float16.BYTES)

    val j: Int
    val nibble: Int
    val xh: Int
    if (inBlockIndex < GGMLType.Q5_1.getBlockSize() / 2) {
      j = inBlockIndex
      nibble = Byte.toUnsignedInt(
        FloatTensor.Companion.readByte(
          memorySegment,
          blockOffset + 2L * Float16.BYTES + Integer.BYTES + j
        )
      ) and 0x0F
      xh = ((qh shr j) shl 4) and 0x10
    } else {
      j = inBlockIndex - GGMLType.Q5_1.getBlockSize() / 2
      nibble = (Byte.toUnsignedInt(
        FloatTensor.Companion.readByte(
          memorySegment,
          blockOffset + 2L * Float16.BYTES + Integer.BYTES + j
        )
      ) ushr 4) and 0x0F
      xh = (qh shr (j + 12)) and 0x10
    }

    val q = nibble or xh
    return q * d + m
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor) {
      if (FloatTensor.Companion.USE_VECTOR_API) {
        return vectorDot(this, thisOffset, that, thatOffset, size)
      }
      return scalarDot(this, thisOffset, that, thatOffset, size)
    }
    return FloatTensor.Companion.scalarDot(this, thisOffset, that, thatOffset, size)
  }

  companion object {
    private fun vectorDot(
      thiz: Q5_1FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      assert(Integer.bitCount(GGMLType.Q5_1.getBlockSize()) == 1) { "power of 2" }
      var j = 0
      var result = 0f

      val alignmentBound = min(size, -thisOffset and (GGMLType.Q5_1.getBlockSize() - 1))
      if (alignmentBound > 0) {
        result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
        j = alignmentBound
      }

      val decoded = FloatArray(GGMLType.Q5_1.getBlockSize())
      val upperBound = j + (size - j) / GGMLType.Q5_1.getBlockSize() * GGMLType.Q5_1.getBlockSize()
      val vecUpper = Objects.requireNonNull<VectorSpecies<Float>?>(FloatTensor.Companion.F_SPECIES)
        .loopBound(GGMLType.Q5_1.getBlockSize())
      while (j < upperBound) {
        assert((thisOffset + j) % GGMLType.Q5_1.getBlockSize() == 0)
        val blockOffset = (thisOffset + j).toLong() / GGMLType.Q5_1.getBlockSize() * GGMLType.Q5_1.getTypeSize()
        val d: Float = FloatTensor.Companion.readFloat16(thiz.memorySegment, blockOffset)
        val m: Float = FloatTensor.Companion.readFloat16(thiz.memorySegment, blockOffset + Float16.BYTES)
        val qh: Int = readInt32LE(thiz.memorySegment, blockOffset + 2L * Float16.BYTES)
        val qsBase = blockOffset + 2L * Float16.BYTES + Integer.BYTES

        for (p in 0..<GGMLType.Q5_1.getBlockSize() / 2) {
          val packed = Byte.toUnsignedInt(FloatTensor.Companion.readByte(thiz.memorySegment, qsBase + p))
          val x0 = (packed and 0x0F) or ((((qh shr p) shl 4) and 0x10))
          val x1 = ((packed ushr 4) and 0x0F) or ((qh shr (p + 12)) and 0x10)
          decoded[p] = x0 * d + m
          decoded[p + GGMLType.Q5_1.getBlockSize() / 2] = x1 * d + m
        }

        var acc = FloatVector.zero(FloatTensor.Companion.F_SPECIES)
        run {
          var i = 0
          while (i < vecUpper) {
            val w = FloatVector.fromArray(FloatTensor.Companion.F_SPECIES, decoded, i)
            val x = that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j + i)
            acc = w.fma(x, acc)
            i += FloatTensor.Companion.F_SPECIES.length()
          }
        }
        result += acc.reduceLanes(VectorOperators.ADD)

        for (i in vecUpper..<GGMLType.Q5_1.getBlockSize()) {
          result += decoded[i] * that.values[thatOffset + j + i]
        }
        j += GGMLType.Q5_1.getBlockSize()
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
      val b0 = Byte.toUnsignedInt(FloatTensor.Companion.readByte(memorySegment, offset))
      val b1 = Byte.toUnsignedInt(FloatTensor.Companion.readByte(memorySegment, offset + 1))
      val b2 = Byte.toUnsignedInt(FloatTensor.Companion.readByte(memorySegment, offset + 2))
      val b3 = Byte.toUnsignedInt(FloatTensor.Companion.readByte(memorySegment, offset + 3))
      return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }
  }
}
