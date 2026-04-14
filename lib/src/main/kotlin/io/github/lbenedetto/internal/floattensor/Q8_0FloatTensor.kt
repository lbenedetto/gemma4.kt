package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.F_SPECIES
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.USE_VECTOR_API
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.readByte
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.readFloat16
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.scalarDot
import io.github.lbenedetto.internal.gguf.GGMLType
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder
import kotlin.math.min

internal class Q8_0FloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor {

  override fun get(index: Long): Float {
    val blockIndex = index / GGMLType.Q8_0.blockSize
    val withinBlockIndex = index % GGMLType.Q8_0.blockSize
    val blockOffset = blockIndex * GGMLType.Q8_0.typeSize
    val quant: Byte = readByte(memorySegment, blockOffset + Float16.BYTES + withinBlockIndex)
    val scale: Float = readFloat16(memorySegment, blockOffset)
    return quant * scale
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (USE_VECTOR_API) {
      vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
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

      assert(Integer.bitCount(GGMLType.Q8_0.blockSize) == 1) { "power of 2" }
      val alignmentBound = min(size, -thisOffset and (GGMLType.Q8_0.blockSize - 1))
      if (alignmentBound > 0) {
        result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
        j += alignmentBound
      }
      assert((thisOffset + j) % GGMLType.Q8_0.blockSize == 0)

      var value = FloatVector.zero(F_SPECIES!!)
      var blockOffset = (thisOffset + j).toLong() / GGMLType.Q8_0.blockSize * GGMLType.Q8_0.typeSize
      val upperBound = j + (size - j) / GGMLType.Q8_0.blockSize * GGMLType.Q8_0.blockSize
      while (j < upperBound) {
        val wScaleValue: Float = readFloat16(thiz.memorySegment, blockOffset)
        val wScale = FloatVector.broadcast(F_SPECIES, wScaleValue)
        when (F_SPECIES.vectorBitSize()) {
          512 -> {
            val wBytes = ByteVector.fromMemorySegment(
              ByteVector.SPECIES_256,
              thiz.memorySegment,
              blockOffset + Float16.BYTES,
              ByteOrder.LITTLE_ENDIAN
            )
            val s0 = that.getFloatVector(F_SPECIES, thatOffset + j)
              .mul(wBytes.castShape(F_SPECIES, 0))
            val s1 = that.getFloatVector(
              F_SPECIES,
              thatOffset + j + F_SPECIES.length()
            ).mul(wBytes.castShape(F_SPECIES, 1))
            value = s0.add(s1).fma(wScale, value)
          }

          256 -> {
            val wBytes = ByteVector.fromMemorySegment(
              ByteVector.SPECIES_256,
              thiz.memorySegment,
              blockOffset + Float16.BYTES,
              ByteOrder.LITTLE_ENDIAN
            )
            var s0 = that.getFloatVector(F_SPECIES, thatOffset + j)
              .mul(wBytes.castShape(F_SPECIES, 0))
            var s1 = that.getFloatVector(
              F_SPECIES,
              thatOffset + j + 2 * F_SPECIES.length()
            ).mul(wBytes.castShape(F_SPECIES, 2))
            s0 = that.getFloatVector(
              F_SPECIES,
              thatOffset + j + F_SPECIES.length()
            ).fma(wBytes.castShape(F_SPECIES, 1), s0)
            s1 = that.getFloatVector(
              F_SPECIES,
              thatOffset + j + 3 * F_SPECIES.length()
            ).fma(wBytes.castShape(F_SPECIES, 3), s1)
            value = s0.add(s1).fma(wScale, value)
          }

          128 -> {
            for (i in 0..1) {
              val wBytes = ByteVector.fromMemorySegment(
                ByteVector.SPECIES_128,
                thiz.memorySegment,
                blockOffset + Float16.BYTES + i * ByteVector.SPECIES_128.vectorByteSize(),
                ByteOrder.LITTLE_ENDIAN
              )
              var s0 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16)
                .mul(wBytes.castShape(F_SPECIES, 0))
              var s1 = that.getFloatVector(
                F_SPECIES,
                thatOffset + j + i * 16 + 2 * F_SPECIES.length()
              ).mul(wBytes.castShape(F_SPECIES, 2))
              s0 = that.getFloatVector(
                F_SPECIES,
                thatOffset + j + i * 16 + F_SPECIES.length()
              ).fma(wBytes.castShape(F_SPECIES, 1), s0)
              s1 = that.getFloatVector(
                F_SPECIES,
                thatOffset + j + i * 16 + 3 * F_SPECIES.length()
              ).fma(wBytes.castShape(F_SPECIES, 3), s1)
              value = s0.add(s1).fma(wScale, value)
            }
          }

          else -> throw UnsupportedOperationException(F_SPECIES.toString())
        }
        j += GGMLType.Q8_0.blockSize
        blockOffset += GGMLType.Q8_0.typeSize.toLong()
      }
      result += value.reduceLanes(VectorOperators.ADD)

      if (j < size) {
        result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
      }

      return result
    }
  }
}
