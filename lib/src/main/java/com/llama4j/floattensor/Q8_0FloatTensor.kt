package com.llama4j.floattensor

import com.llama4j.gguf.GGMLType
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder
import java.util.*
import kotlin.math.min

internal class Q8_0FloatTensor(size: Long, memorySegment: MemorySegment) : FloatTensor() {
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
    return GGMLType.Q8_0
  }

  override fun getFloat(index: Long): Float {
    val blockIndex = index / GGMLType.Q8_0.getBlockSize()
    val withinBlockIndex = index % GGMLType.Q8_0.getBlockSize()
    val blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize()
    val quant: Byte = FloatTensor.Companion.readByte(memorySegment, blockOffset + Float16.BYTES + withinBlockIndex)
    val scale: Float = FloatTensor.Companion.readFloat16(memorySegment, blockOffset)
    return quant * scale
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
      thiz: Q8_0FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      var result = 0f
      var j = 0

      assert(Integer.bitCount(GGMLType.Q8_0.getBlockSize()) == 1) { "power of 2" }
      val alignmentBound = min(size, -thisOffset and (GGMLType.Q8_0.getBlockSize() - 1))
      if (alignmentBound > 0) {
        result += FloatTensor.Companion.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
        j += alignmentBound
      }
      assert((thisOffset + j) % GGMLType.Q8_0.getBlockSize() == 0)

      var `val` = FloatVector.zero(Objects.requireNonNull<VectorSpecies<Float>?>(FloatTensor.Companion.F_SPECIES))
      var blockOffset = (thisOffset + j).toLong() / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getTypeSize()
      val upperBound = j + (size - j) / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getBlockSize()
      while (j < upperBound) {
        val wScaleValue: Float = FloatTensor.Companion.readFloat16(thiz.memorySegment, blockOffset)
        val wScale = FloatVector.broadcast(FloatTensor.Companion.F_SPECIES, wScaleValue)
        when (FloatTensor.Companion.F_SPECIES.vectorBitSize()) {
          512 -> {
            val wBytes = ByteVector.fromMemorySegment(
              ByteVector.SPECIES_256,
              thiz.memorySegment,
              blockOffset + Float16.BYTES,
              ByteOrder.LITTLE_ENDIAN
            )
            val s0 = that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j)
              .mul(wBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
            val s1 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + FloatTensor.Companion.F_SPECIES.length()
            ).mul(wBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 1))
            `val` = s0.add(s1).fma(wScale, `val`)
          }

          256 -> {
            val wBytes = ByteVector.fromMemorySegment(
              ByteVector.SPECIES_256,
              thiz.memorySegment,
              blockOffset + Float16.BYTES,
              ByteOrder.LITTLE_ENDIAN
            )
            var s0 = that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j)
              .mul(wBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
            var s1 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + 2 * FloatTensor.Companion.F_SPECIES.length()
            ).mul(wBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 2))
            s0 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + FloatTensor.Companion.F_SPECIES.length()
            ).fma(wBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 1), s0)
            s1 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + 3 * FloatTensor.Companion.F_SPECIES.length()
            ).fma(wBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 3), s1)
            `val` = s0.add(s1).fma(wScale, `val`)
          }

          128 -> {
            for (i in 0..1) {
              val wBytes = ByteVector.fromMemorySegment(
                ByteVector.SPECIES_128,
                thiz.memorySegment,
                blockOffset + Float16.BYTES + i * ByteVector.SPECIES_128.vectorByteSize(),
                ByteOrder.LITTLE_ENDIAN
              )
              var s0 = that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j + i * 16)
                .mul(wBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
              var s1 = that.getFloatVector(
                FloatTensor.Companion.F_SPECIES,
                thatOffset + j + i * 16 + 2 * FloatTensor.Companion.F_SPECIES.length()
              ).mul(wBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 2))
              s0 = that.getFloatVector(
                FloatTensor.Companion.F_SPECIES,
                thatOffset + j + i * 16 + FloatTensor.Companion.F_SPECIES.length()
              ).fma(wBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 1), s0)
              s1 = that.getFloatVector(
                FloatTensor.Companion.F_SPECIES,
                thatOffset + j + i * 16 + 3 * FloatTensor.Companion.F_SPECIES.length()
              ).fma(wBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 3), s1)
              `val` = s0.add(s1).fma(wScale, `val`)
            }
          }

          else -> throw UnsupportedOperationException(FloatTensor.Companion.F_SPECIES.toString())
        }
        j += GGMLType.Q8_0.getBlockSize()
        blockOffset += GGMLType.Q8_0.getTypeSize().toLong()
      }
      result += `val`.reduceLanes(VectorOperators.ADD)

      if (j < size) {
        result += FloatTensor.Companion.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
      }

      return result
    }
  }
}
