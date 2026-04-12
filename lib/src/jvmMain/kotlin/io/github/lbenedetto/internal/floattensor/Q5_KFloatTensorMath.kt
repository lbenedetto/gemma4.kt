package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.Q5_KFloatTensor.Companion.BLOCK_SIZE
import io.github.lbenedetto.internal.floattensor.Q5_KFloatTensor.Companion.TYPE_SIZE
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.F_SPECIES
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import java.nio.ByteOrder
import kotlin.math.min

actual object Q5_KFloatTensorMath {
  internal actual fun vectorDot(
    thiz: Q5_KFloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float {
    var result = 0f
    var j = 0

    assert(Integer.bitCount(BLOCK_SIZE) == 1) { "power of 2" }
    val alignmentBound = min(size, -thisOffset and (BLOCK_SIZE - 1))
    if (alignmentBound > 0) {
      result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
      j += alignmentBound
    }

    var value = FloatVector.zero(F_SPECIES!!)
    var val2 = FloatVector.zero(F_SPECIES)
    var blockOffset: Long = (thisOffset + j).toLong() / BLOCK_SIZE * TYPE_SIZE
    val upperBound: Int = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE

    while (j < upperBound) {
      val d: Float = thiz.memorySegment.readFloat16(blockOffset)
      val dmin: Float = thiz.memorySegment.readFloat16(blockOffset + 2)
      val scalesOff = blockOffset + 4
      val qhOff = blockOffset + 16
      val qsOff = blockOffset + 48
      val qh0 =
        ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment.actual(), qhOff, ByteOrder.LITTLE_ENDIAN)
      val qh1 =
        ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment.actual(), qhOff + 16, ByteOrder.LITTLE_ENDIAN)

      for (g in 0..3) {
        val loSubBlock = g * 2
        val hiSubBlock = loSubBlock + 1
        val d1: Float = d * Q4_KFloatTensor.getScaleMinK4(loSubBlock, thiz.memorySegment, scalesOff, false)
        val m1: Float =
          dmin * Q4_KFloatTensor.getScaleMinK4(loSubBlock, thiz.memorySegment, scalesOff, true)
        val d2: Float = d * Q4_KFloatTensor.getScaleMinK4(hiSubBlock, thiz.memorySegment, scalesOff, false)
        val m2: Float =
          dmin * Q4_KFloatTensor.getScaleMinK4(hiSubBlock, thiz.memorySegment, scalesOff, true)
        val qhBitPosLo = 2 * g
        val qhBitPosHi = qhBitPosLo + 1
        val groupQsOff = qsOff + g.toLong() * 32
        val d1Vec = FloatVector.broadcast(F_SPECIES, d1)
        val d2Vec = FloatVector.broadcast(F_SPECIES, d2)
        val negM1Vec = FloatVector.broadcast(F_SPECIES, -m1)
        val negM2Vec = FloatVector.broadcast(F_SPECIES, -m2)

        for (c in 0..1) {
          val loBase = thatOffset + j + g * 64 + c * 16
          val hiBase = thatOffset + j + g * 64 + 32 + c * 16

          val wBytes = ByteVector.fromMemorySegment(
            ByteVector.SPECIES_128, thiz.memorySegment.actual(),
            groupQsOff + c * 16L, ByteOrder.LITTLE_ENDIAN
          )
          var loQ = wBytes.and(0xF.toByte())
          var hiQ: ByteVector = wBytes.lanewise(VectorOperators.LSHR, 4L)

          val qhBytes = if (c == 0) qh0 else qh1
          loQ = loQ.or(
            qhBytes.lanewise(VectorOperators.LSHR, qhBitPosLo.toLong()).and(1.toByte())
              .lanewise(VectorOperators.LSHL, 4L)
          )
          hiQ = hiQ.or(
            qhBytes.lanewise(VectorOperators.LSHR, qhBitPosHi.toLong()).and(1.toByte())
              .lanewise(VectorOperators.LSHL, 4L)
          )

          when (F_SPECIES.vectorBitSize()) {
            512 -> {
              val loQf = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats()
              val hiQf = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats()
              value =
                loQf.fma(d1Vec, negM1Vec).fma(that.getFloatVector(loBase), value)
              val2 = hiQf.fma(d2Vec, negM2Vec).fma(that.getFloatVector(hiBase), val2)
            }

            256 -> {
              val loQf0 = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats()
              val loQf1 = loQ.castShape(F_SPECIES, 1).reinterpretAsFloats()
              val hiQf0 = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats()
              val hiQf1 = hiQ.castShape(F_SPECIES, 1).reinterpretAsFloats()
              value =
                loQf0.fma(d1Vec, negM1Vec).fma(that.getFloatVector(loBase), value)
              value = loQf1.fma(d1Vec, negM1Vec)
                .fma(that.getFloatVector(loBase + F_SPECIES.length()), value)
              val2 =
                hiQf0.fma(d2Vec, negM2Vec).fma(that.getFloatVector(hiBase), val2)
              val2 = hiQf1.fma(d2Vec, negM2Vec)
                .fma(that.getFloatVector(hiBase + F_SPECIES.length()), val2)
            }

            128 -> {
              for (p in 0..3) {
                val off: Int = p * F_SPECIES.length()
                val loQf = loQ.castShape(F_SPECIES, p).reinterpretAsFloats()
                val hiQf = hiQ.castShape(F_SPECIES, p).reinterpretAsFloats()
                value = loQf.fma(d1Vec, negM1Vec)
                  .fma(that.getFloatVector(loBase + off), value)
                val2 = hiQf.fma(d2Vec, negM2Vec)
                  .fma(that.getFloatVector(hiBase + off), val2)
              }
            }

            else -> throw UnsupportedOperationException(F_SPECIES.toString())
          }
        }
      }
      j += BLOCK_SIZE
      blockOffset += TYPE_SIZE.toLong()
    }

    result += value.add(val2).reduceLanes(VectorOperators.ADD)

    if (j < size) {
      result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
    }

    return result
  }

}
