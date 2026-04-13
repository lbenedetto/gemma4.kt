package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.toUnsignedInt
import io.github.lbenedetto.internal.floattensor.Q5_1FloatTensor.Companion.readInt32LE
import io.github.lbenedetto.internal.floattensor.Q5_1FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.F_SPECIES
import io.github.lbenedetto.internal.gguf.GGMLType
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import kotlin.math.min

object Q5_1FloatTensorMath {
  internal fun vectorDot(
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
      val d: Float = thiz.memorySegment.readFloat16(blockOffset)
      val m: Float = thiz.memorySegment.readFloat16(blockOffset + Float16.BYTES)
      val qh: Int = readInt32LE(thiz.memorySegment, blockOffset + 2L * Float16.BYTES)
      val qsBase = blockOffset + 2L * Float16.BYTES + Integer.BYTES

      for (p in 0..<GGMLType.Q5_1.blockSize / 2) {
        val packed = thiz.memorySegment.readByte(qsBase + p).toUnsignedInt()
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
          val x = that.getFloatVector(thatOffset + j + i)
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

}
