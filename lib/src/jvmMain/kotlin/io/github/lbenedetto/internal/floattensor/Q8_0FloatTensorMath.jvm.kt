package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.F_SPECIES
import io.github.lbenedetto.internal.gguf.GGMLType
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import java.nio.ByteOrder
import kotlin.math.min

actual object Q8_0FloatTensorMath {
  internal actual fun vectorDot(
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

    var `val` = FloatVector.zero(F_SPECIES!!)
    var blockOffset = (thisOffset + j).toLong() / GGMLType.Q8_0.blockSize * GGMLType.Q8_0.typeSize
    val upperBound = j + (size - j) / GGMLType.Q8_0.blockSize * GGMLType.Q8_0.blockSize
    while (j < upperBound) {
      val wScaleValue: Float = thiz.memorySegment.readFloat16(blockOffset)
      val wScale = FloatVector.broadcast(F_SPECIES, wScaleValue)
      when (F_SPECIES.vectorBitSize()) {
        512 -> {
          val wBytes = ByteVector.fromMemorySegment(
            ByteVector.SPECIES_256,
            thiz.memorySegment.actual(),
            blockOffset + Float16.BYTES,
            ByteOrder.LITTLE_ENDIAN
          )
          val s0 = that.getFloatVector(thatOffset + j)
            .mul(wBytes.castShape(F_SPECIES, 0))
          val s1 = that.getFloatVector(thatOffset + j + F_SPECIES.length())
            .mul(wBytes.castShape(F_SPECIES, 1))
          `val` = s0.add(s1).fma(wScale, `val`)
        }

        256 -> {
          val wBytes = ByteVector.fromMemorySegment(
            ByteVector.SPECIES_256,
            thiz.memorySegment.actual(),
            blockOffset + Float16.BYTES,
            ByteOrder.LITTLE_ENDIAN
          )
          var s0 = that.getFloatVector(thatOffset + j)
            .mul(wBytes.castShape(F_SPECIES, 0))
          var s1 = that.getFloatVector(thatOffset + j + 2 * F_SPECIES.length())
            .mul(wBytes.castShape(F_SPECIES, 2))
          s0 = that.getFloatVector(thatOffset + j + F_SPECIES.length())
            .fma(wBytes.castShape(F_SPECIES, 1), s0)
          s1 = that.getFloatVector(thatOffset + j + 3 * F_SPECIES.length())
            .fma(wBytes.castShape(F_SPECIES, 3), s1)
          `val` = s0.add(s1).fma(wScale, `val`)
        }

        128 -> {
          for (i in 0..1) {
            val wBytes = ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128,
              thiz.memorySegment.actual(),
              blockOffset + Float16.BYTES + i * ByteVector.SPECIES_128.vectorByteSize(),
              ByteOrder.LITTLE_ENDIAN
            )
            var s0 = that.getFloatVector(thatOffset + j + i * 16)
              .mul(wBytes.castShape(F_SPECIES, 0))
            var s1 = that.getFloatVector(thatOffset + j + i * 16 + 2 * F_SPECIES.length())
              .mul(wBytes.castShape(F_SPECIES, 2))
            s0 = that.getFloatVector(thatOffset + j + i * 16 + F_SPECIES.length())
              .fma(wBytes.castShape(F_SPECIES, 1), s0)
            s1 = that.getFloatVector(thatOffset + j + i * 16 + 3 * F_SPECIES.length())
              .fma(wBytes.castShape(F_SPECIES, 3), s1)
            `val` = s0.add(s1).fma(wScale, `val`)
          }
        }

        else -> throw UnsupportedOperationException(F_SPECIES.toString())
      }
      j += GGMLType.Q8_0.blockSize
      blockOffset += GGMLType.Q8_0.typeSize.toLong()
    }
    result += `val`.reduceLanes(VectorOperators.ADD)

    if (j < size) {
      result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
    }

    return result
  }

}
