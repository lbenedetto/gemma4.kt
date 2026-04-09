package com.llama4j.floattensor

import com.llama4j.gguf.GGMLType
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.Byte
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder
import java.util.*
import kotlin.Float
import kotlin.Int
import kotlin.Long
import kotlin.UnsupportedOperationException
import kotlin.assert
import kotlin.collections.minus
import kotlin.collections.plus
import kotlin.collections.plusAssign
import kotlin.math.min
import kotlin.plus
import kotlin.sequences.minus
import kotlin.sequences.plus
import kotlin.text.toInt
import kotlin.text.toLong
import kotlin.times
import kotlin.toString

internal class Q4_1FloatTensor(size: Long, memorySegment: MemorySegment) : FloatTensor() {
  val size: Long
  val memorySegment: MemorySegment

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
    return GGMLType.Q4_1
  }

  override fun getFloat(index: Long): Float {
    assert(0 <= index && index < size)
    val blockIndex = index / GGMLType.Q4_1.getBlockSize()
    val blockOffset = blockIndex * GGMLType.Q4_1.getTypeSize()
    val delta: Float = FloatTensor.Companion.readFloat16(memorySegment, blockOffset)
    val min: Float = FloatTensor.Companion.readFloat16(memorySegment, blockOffset + Float16.BYTES)
    val modIndex = (index % GGMLType.Q4_1.getBlockSize()).toInt()
    val quant: Int
    if (modIndex < 16) {
      quant = Byte.toUnsignedInt(
        FloatTensor.Companion.readByte(
          memorySegment,
          blockOffset + 2 * Float16.BYTES + modIndex
        )
      ) and 0x0F
    } else {
      quant = (Byte.toUnsignedInt(
        FloatTensor.Companion.readByte(
          memorySegment,
          blockOffset + 2 * Float16.BYTES + modIndex - 16
        )
      ) ushr 4) and 0x0F
    }
    return delta * quant + min
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (FloatTensor.Companion.USE_VECTOR_API) {
      return vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      return FloatTensor.Companion.scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }

  companion object {
    private fun vectorDot(
      thiz: Q4_1FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      var result = 0f
      var j = 0

      assert(Integer.bitCount(GGMLType.Q4_1.getBlockSize()) == 1) { "power of 2" }
      val alignmentBound = min(size, -thisOffset and (GGMLType.Q4_1.getBlockSize() - 1))
      if (alignmentBound > 0) {
        result += FloatTensor.Companion.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
        j += alignmentBound
      }
      assert((thisOffset + j) % GGMLType.Q4_1.getBlockSize() == 0)

      var `val` = FloatVector.zero(Objects.requireNonNull<VectorSpecies<Float>?>(FloatTensor.Companion.F_SPECIES))
      var blockOffset = (thisOffset + j).toLong() / GGMLType.Q4_1.getBlockSize() * GGMLType.Q4_1.getTypeSize()
      val upperBound = j + (size - j) / GGMLType.Q4_1.getBlockSize() * GGMLType.Q4_1.getBlockSize()
      while (j < upperBound) {
        val deltaValue: Float = FloatTensor.Companion.readFloat16(thiz.memorySegment, blockOffset)
        val minValue: Float = FloatTensor.Companion.readFloat16(thiz.memorySegment, blockOffset + Float16.BYTES)
        val wDelta = FloatVector.broadcast(FloatTensor.Companion.F_SPECIES, deltaValue)
        val wMin = FloatVector.broadcast(FloatTensor.Companion.F_SPECIES, minValue)
        val wBytes = ByteVector.fromMemorySegment(
          ByteVector.SPECIES_128,
          thiz.memorySegment,
          blockOffset + 2 * Float16.BYTES,
          ByteOrder.LITTLE_ENDIAN
        )
        val loBytes = wBytes.and(0xF.toByte())
        val hiBytes: ByteVector = wBytes.lanewise(VectorOperators.LSHR, 4)
        when (FloatTensor.Companion.F_SPECIES.vectorBitSize()) {
          512 -> {
            val that0 = that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j)
            val that1 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + FloatTensor.Companion.F_SPECIES.length()
            )
            val s0 = that0.mul(loBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
            val s1 = that1.mul(hiBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
            `val` = s0.add(s1).fma(wDelta, `val`)
            `val` = that0.add(that1).fma(wMin, `val`)
          }

          256 -> {
            val that0 = that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j)
            val that1 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + FloatTensor.Companion.F_SPECIES.length()
            )
            val that2 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + 2 * FloatTensor.Companion.F_SPECIES.length()
            )
            val that3 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + 3 * FloatTensor.Companion.F_SPECIES.length()
            )
            var s0 = that0.mul(loBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
            var s1 = that2.mul(hiBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
            s0 = that1.fma(loBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 1), s0)
            s1 = that3.fma(hiBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 1), s1)
            `val` = s0.add(s1).fma(wDelta, `val`)
            `val` = that0.add(that1).add(that2).add(that3).fma(wMin, `val`)
          }

          128 -> {
            for (i in 0..1) {
              val tmp = if (i == 0) loBytes else hiBytes
              var s0 = that.getFloatVector(
                FloatTensor.Companion.F_SPECIES,
                thatOffset + j + (i * 4) * FloatTensor.Companion.F_SPECIES.length()
              ).mul(tmp.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
              var s1 = that.getFloatVector(
                FloatTensor.Companion.F_SPECIES,
                thatOffset + j + (i * 4 + 2) * FloatTensor.Companion.F_SPECIES.length()
              ).mul(tmp.castShape<Float>(FloatTensor.Companion.F_SPECIES, 2))
              s0 = that.getFloatVector(
                FloatTensor.Companion.F_SPECIES,
                thatOffset + j + (i * 4 + 1) * FloatTensor.Companion.F_SPECIES.length()
              ).fma(tmp.castShape<Float>(FloatTensor.Companion.F_SPECIES, 1), s0)
              s1 = that.getFloatVector(
                FloatTensor.Companion.F_SPECIES,
                thatOffset + j + (i * 4 + 3) * FloatTensor.Companion.F_SPECIES.length()
              ).fma(tmp.castShape<Float>(FloatTensor.Companion.F_SPECIES, 3), s1)
              `val` = s0.add(s1).fma(wDelta, `val`)
            }
            // vectorized min contribution
            var thatSum = FloatVector.zero(FloatTensor.Companion.F_SPECIES)
            var k = 0
            while (k < GGMLType.Q4_1.getBlockSize()) {
              thatSum = thatSum.add(that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j + k))
              k += FloatTensor.Companion.F_SPECIES.length()
            }
            `val` = thatSum.fma(wMin, `val`)
          }

          else -> throw UnsupportedOperationException(FloatTensor.Companion.F_SPECIES.toString())
        }
        j += GGMLType.Q4_1.getBlockSize()
        blockOffset += GGMLType.Q4_1.getTypeSize().toLong()
      }
      result += `val`.reduceLanes(VectorOperators.ADD)

      if (j < size) {
        result += FloatTensor.Companion.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
      }

      return result
    }
  }
}
