package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.Q4_KFloatTensor.Companion.BLOCK_SIZE
import io.github.lbenedetto.internal.floattensor.Q4_KFloatTensor.Companion.TYPE_SIZE
import io.github.lbenedetto.internal.floattensor.Q4_KFloatTensor.Companion.getScaleMinK4
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.F_SPECIES
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import java.nio.ByteOrder
import kotlin.math.min

actual object Q4_KFloatTensorMath {
  internal actual fun vectorDot(
    thiz: Q4_KFloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float {
    var result = 0f
    var j = 0

    // Handle unaligned head
    assert(Integer.bitCount(BLOCK_SIZE) == 1) { "power of 2" }
    val alignmentBound = min(size, -thisOffset and (BLOCK_SIZE - 1))
    if (alignmentBound > 0) {
      result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
      j += alignmentBound
    }

    var value1 = FloatVector.zero(F_SPECIES!!)
    var value2 = FloatVector.zero(F_SPECIES)
    var blockOffset: Long = (thisOffset + j).toLong() / BLOCK_SIZE * TYPE_SIZE
    val upperBound: Int = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE

    while (j < upperBound) {
      val d: Float = thiz.memorySegment.readFloat16(blockOffset)
      val dmin: Float = thiz.memorySegment.readFloat16(blockOffset + 2)
      val scalesOff = blockOffset + 4
      val qsOff = blockOffset + 16

      // 4 groups of 64 values each (2 sub-blocks per group: low nibble + high nibble)
      for (g in 0..3) {
        val d1: Float = d * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, false)
        val negM1: Float = -(dmin * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, true))
        val d2: Float = d * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, false)
        val negM2: Float = -(dmin * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, true))

        val d1Vec = FloatVector.broadcast(F_SPECIES, d1)
        val negM1Vec = FloatVector.broadcast(F_SPECIES, negM1)
        val d2Vec = FloatVector.broadcast(F_SPECIES, d2)
        val negM2Vec = FloatVector.broadcast(F_SPECIES, negM2)

        val loBase = thatOffset + j + g * 64
        val hiBase = thatOffset + j + g * 64 + 32

        // Process 32 bytes of qs in 2 chunks of 16 bytes
        for (c in 0..1) {
          val wBytes = ByteVector.fromMemorySegment(
            ByteVector.SPECIES_128, thiz.memorySegment.actual(),
            qsOff + g.toLong() * 32 + c * 16, ByteOrder.LITTLE_ENDIAN
          )
          val loBytes = wBytes.and(0xF.toByte())
          val hiBytes: ByteVector = wBytes.lanewise(VectorOperators.LSHR, 4L)

          val loIdx = loBase + c * 16
          val hiIdx = hiBase + c * 16

          when (F_SPECIES.vectorBitSize()) {
            512 -> {
              val loQ = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats()
              val hiQ = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats()
              value1 = loQ.fma(d1Vec, negM1Vec)
                .fma(that.getFloatVector(loIdx), value1)
              value2 = hiQ.fma(d2Vec, negM2Vec)
                .fma(that.getFloatVector(hiIdx), value2)
            }

            256 -> {
              val loQ0 = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats()
              val loQ1 = loBytes.castShape(F_SPECIES, 1).reinterpretAsFloats()
              value1 = loQ0.fma(d1Vec, negM1Vec)
                .fma(that.getFloatVector(loIdx), value1)
              value2 = loQ1.fma(d1Vec, negM1Vec)
                .fma(that.getFloatVector(loIdx + F_SPECIES.length()), value2)
              val hiQ0 = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats()
              val hiQ1 = hiBytes.castShape(F_SPECIES, 1).reinterpretAsFloats()
              value1 = hiQ0.fma(d2Vec, negM2Vec)
                .fma(that.getFloatVector(hiIdx), value1)
              value2 = hiQ1.fma(d2Vec, negM2Vec)
                .fma(that.getFloatVector(hiIdx + F_SPECIES.length()), value2)
            }

            128 -> {
              for (p in 0..3) {
                val loQ = loBytes.castShape(F_SPECIES, p).reinterpretAsFloats()
                value1 = loQ.fma(d1Vec, negM1Vec)
                  .fma(that.getFloatVector(loIdx + p * F_SPECIES.length()), value1)
                val hiQ = hiBytes.castShape(F_SPECIES, p).reinterpretAsFloats()
                value2 = hiQ.fma(d2Vec, negM2Vec)
                  .fma(that.getFloatVector(hiIdx + p * F_SPECIES.length()), value2)
              }
            }

            else -> throw UnsupportedOperationException(F_SPECIES.toString())
          }
        }
      }
      j += BLOCK_SIZE
      blockOffset += TYPE_SIZE.toLong()
    }
    result += value1.add(value2).reduceLanes(VectorOperators.ADD)

    // Handle tail
    if (j < size) {
      result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
    }

    return result
  }

}
