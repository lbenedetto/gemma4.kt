package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.F_SPECIES
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.USE_VECTOR_API
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.readByte
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.readFloat16
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.scalarDot
import io.github.lbenedetto.internal.floattensor.FloatTensorHelpers.toUnsignedInt
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.gguf.QK_K
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder
import kotlin.math.min

internal class Q5_KFloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor {

  override fun get(index: Long): Float {
    val blockIndex: Long = index / BLOCK_SIZE
    val withinBlock = (index % BLOCK_SIZE).toInt()
    val blockOffset: Long = blockIndex * TYPE_SIZE
    val d: Float = readFloat16(memorySegment, blockOffset)
    val dmin: Float = readFloat16(memorySegment, blockOffset + 2)
    val scalesOffset = blockOffset + 4
    val qhOffset = blockOffset + 16 // 4 + 12
    val qsOffset = blockOffset + 48 // 4 + 12 + 32

    val group = withinBlock / 64
    val inGroup = withinBlock % 64
    val isHigh = inGroup >= 32
    val l = if (isHigh) inGroup - 32 else inGroup
    val subBlock = if (isHigh) group * 2 + 1 else group * 2

    val sc: Int = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, false)
    val m: Int = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, true)

    val qsByte: Byte = readByte(memorySegment, qsOffset + group * 32 + l)
    val nibble = if (isHigh) ((qsByte.toUnsignedInt() shr 4) and 0xF) else (qsByte.toUnsignedInt() and 0xF)

    val qhBitPos = if (isHigh) 2 * group + 1 else 2 * group
    val qhBit = (readByte(memorySegment, qhOffset + l).toUnsignedInt() shr qhBitPos) and 1

    val quant = nibble or (qhBit shl 4)
    return d * sc * quant - dmin * m
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (USE_VECTOR_API) {
      vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }

  companion object {
    const val BLOCK_SIZE: Int = QK_K
    val TYPE_SIZE: Int = GGMLType.Q5_K.typeSize

    private fun vectorDot(
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
        val d: Float = readFloat16(thiz.memorySegment, blockOffset)
        val dmin: Float = readFloat16(thiz.memorySegment, blockOffset + 2)
        val scalesOff = blockOffset + 4
        val qhOff = blockOffset + 16
        val qsOff = blockOffset + 48
        val qh0 =
          ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, qhOff, ByteOrder.LITTLE_ENDIAN)
        val qh1 =
          ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, qhOff + 16, ByteOrder.LITTLE_ENDIAN)

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
              ByteVector.SPECIES_128, thiz.memorySegment,
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
                  loQf.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase), value)
                val2 = hiQf.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase), val2)
              }

              256 -> {
                val loQf0 = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats()
                val loQf1 = loQ.castShape(F_SPECIES, 1).reinterpretAsFloats()
                val hiQf0 = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats()
                val hiQf1 = hiQ.castShape(F_SPECIES, 1).reinterpretAsFloats()
                value =
                  loQf0.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase), value)
                value = loQf1.fma(d1Vec, negM1Vec).fma(
                  that.getFloatVector(
                    F_SPECIES,
                    loBase + F_SPECIES.length()
                  ), value
                )
                val2 =
                  hiQf0.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase), val2)
                val2 = hiQf1.fma(d2Vec, negM2Vec).fma(
                  that.getFloatVector(
                    F_SPECIES,
                    hiBase + F_SPECIES.length()
                  ), val2
                )
              }

              128 -> {
                for (p in 0..3) {
                  val off: Int = p * F_SPECIES.length()
                  val loQf = loQ.castShape(F_SPECIES, p).reinterpretAsFloats()
                  val hiQf = hiQ.castShape(F_SPECIES, p).reinterpretAsFloats()
                  value = loQf.fma(d1Vec, negM1Vec)
                    .fma(that.getFloatVector(F_SPECIES, loBase + off), value)
                  val2 = hiQf.fma(d2Vec, negM2Vec)
                    .fma(that.getFloatVector(F_SPECIES, hiBase + off), val2)
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
}
