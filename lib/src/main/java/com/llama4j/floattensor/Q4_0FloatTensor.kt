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

internal class Q4_0FloatTensor(size: Long, memorySegment: MemorySegment) : FloatTensor() {
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
    return GGMLType.Q4_0
  }

  override fun getFloat(index: Long): Float {
    assert(0 <= index && index < size)
    val blockIndex = index / GGMLType.Q4_0.getBlockSize()
    val blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize()
    val scale: Float = FloatTensor.Companion.readFloat16(memorySegment, blockOffset)
    var quant: Byte
    val modIndex = (index % GGMLType.Q4_0.getBlockSize()).toInt()
    if (modIndex < GGMLType.Q4_0.getBlockSize() / 2) {
      quant = (FloatTensor.Companion.readByte(memorySegment, blockOffset + Float16.BYTES + modIndex)
        .toInt() and 0x0F).toByte()
    } else {
      quant = ((FloatTensor.Companion.readByte(
        memorySegment,
        blockOffset + Float16.BYTES + modIndex - GGMLType.Q4_0.getBlockSize() / 2
      ).toInt() ushr 4) and 0x0F).toByte()
    }
    quant = (quant - 8).toByte()
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
      thiz: Q4_0FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      var result = 0f
      var j = 0

      assert(Integer.bitCount(GGMLType.Q4_0.getBlockSize()) == 1) { "power of 2" }
      val alignmentBound = min(size, -thisOffset and (GGMLType.Q4_0.getBlockSize() - 1))
      if (alignmentBound > 0) {
        result += FloatTensor.Companion.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
        j += alignmentBound
      }
      assert((thisOffset + j) % GGMLType.Q4_0.getBlockSize() == 0)

      var `val` = FloatVector.zero(Objects.requireNonNull<VectorSpecies<Float>?>(FloatTensor.Companion.F_SPECIES))
      var blockOffset = (thisOffset + j).toLong() / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getTypeSize()
      val upperBound = j + (size - j) / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getBlockSize()
      while (j < upperBound) {
        val wScaleValue: Float = FloatTensor.Companion.readFloat16(thiz.memorySegment, blockOffset)
        val wScale = FloatVector.broadcast(FloatTensor.Companion.F_SPECIES, wScaleValue)
        val wBytes = ByteVector.fromMemorySegment(
          ByteVector.SPECIES_128,
          thiz.memorySegment,
          blockOffset + Float16.BYTES,
          ByteOrder.LITTLE_ENDIAN
        )
        val loBytes = wBytes.and(0xF.toByte()).sub(8.toByte())
        val hiBytes: ByteVector = wBytes.lanewise(VectorOperators.LSHR, 4).sub(8.toByte())
        when (FloatTensor.Companion.F_SPECIES.vectorBitSize()) {
          512 -> {
            val s0 = that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j)
              .mul(loBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
            val s1 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + FloatTensor.Companion.F_SPECIES.length()
            ).mul(hiBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
            `val` = s0.add(s1).fma(wScale, `val`)
          }

          256 -> {
            var s0 = that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j)
              .mul(loBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
            var s1 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + 2 * FloatTensor.Companion.F_SPECIES.length()
            ).mul(hiBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0))
            s0 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + FloatTensor.Companion.F_SPECIES.length()
            ).fma(loBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 1), s0)
            s1 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + 3 * FloatTensor.Companion.F_SPECIES.length()
            ).fma(hiBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 1), s1)
            `val` = s0.add(s1).fma(wScale, `val`)
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
              `val` = s0.add(s1).fma(wScale, `val`)
            }
          }

          else -> throw UnsupportedOperationException(FloatTensor.Companion.F_SPECIES.toString())
        }
        j += GGMLType.Q4_0.getBlockSize()
        blockOffset += GGMLType.Q4_0.getTypeSize().toLong()
      }
      result += `val`.reduceLanes(VectorOperators.ADD)

      if (j < size) {
        result += FloatTensor.Companion.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
      }

      return result
    }
  }
}
