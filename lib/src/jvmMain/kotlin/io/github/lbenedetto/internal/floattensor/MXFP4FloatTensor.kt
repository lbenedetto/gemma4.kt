package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.gguf.QK_MXFP4
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder
import kotlin.math.min

internal class MXFP4FloatTensor(
  override val size: Long,
  private val memorySegment: MemorySegment
) : FloatTensor() {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector? {
    throw UnsupportedOperationException("getFloatVector")
  }

  override fun type(): GGMLType {
    return GGMLType.MXFP4
  }

  override fun getFloat(index: Long): Float {
    assert(index in 0..<size)
    val blockIndex: Long = index / QK_MXFP4
    val inBlockIndex = (index % QK_MXFP4).toInt()
    val blockOffset = blockIndex * GGMLType.MXFP4.typeSize

    val e8m0 = readByte(memorySegment, blockOffset).toUnsignedInt()
    val d: Float = e8m0ToFp32Half(e8m0)

    val qsOffset = blockOffset + Byte.SIZE_BYTES + (inBlockIndex and 0x0F)
    val packed = readByte(memorySegment, qsOffset).toUnsignedInt()
    val nibble = if (inBlockIndex < (QK_MXFP4 / 2)) (packed and 0x0F) else ((packed ushr 4) and 0x0F)

    return MXFP4_VALUES[nibble] * d
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor) {
      if (USE_VECTOR_API) {
        return vectorDot(this, thisOffset, that, thatOffset, size)
      }
      return scalarDot(this, thisOffset, that, thatOffset, size)
    }
    return scalarDot(this, thisOffset, that, thatOffset, size)
  }

  companion object {
    private val MXFP4_VALUES = intArrayOf(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

    private fun vectorDot(
      thiz: MXFP4FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      assert(Integer.bitCount(QK_MXFP4) == 1) { "power of 2" }
      var j = 0
      var result = 0f

      val alignmentBound = min(size, -thisOffset and (QK_MXFP4 - 1))
      if (alignmentBound > 0) {
        result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
        j = alignmentBound
      }

      val upperBound: Int = j + (size - j) / QK_MXFP4 * QK_MXFP4
      while (j < upperBound) {
        assert((thisOffset + j) % QK_MXFP4 == 0)
        val blockOffset: Long = (thisOffset + j).toLong() / QK_MXFP4 * GGMLType.MXFP4.typeSize
        val d: Float = e8m0ToFp32Half(readByte(thiz.memorySegment, blockOffset).toUnsignedInt())

        val packed = ByteVector.fromMemorySegment(
          ByteVector.SPECIES_128,
          thiz.memorySegment,
          blockOffset + Byte.SIZE_BYTES,
          ByteOrder.LITTLE_ENDIAN
        )
        val lo = packed.and(0x0F.toByte())
        val hi: ByteVector = packed.lanewise(VectorOperators.LSHR, 4L)

        var blockSum = 0f
        when (F_SPECIES!!.vectorBitSize()) {
          512 -> {
            val loCoeffs: FloatVector = mxfp4CodesToCoeffs(
              (lo.castShape(
                F_SPECIES,
                0
              ) as FloatVector?)!!
            )
            val hiCoeffs: FloatVector = mxfp4CodesToCoeffs(
              (hi.castShape(
                F_SPECIES,
                0
              ) as FloatVector?)!!
            )
            val xLo = that.getFloatVector(F_SPECIES, thatOffset + j)
            val xHi =
              that.getFloatVector(F_SPECIES, thatOffset + j + QK_MXFP4 / 2)
            blockSum += loCoeffs.fma(xLo, hiCoeffs.mul(xHi)).reduceLanes(VectorOperators.ADD)
          }

          256 -> {
            val lo0: FloatVector = mxfp4CodesToCoeffs(
              (lo.castShape(F_SPECIES, 0) as FloatVector?)!!
            )
            val lo1: FloatVector = mxfp4CodesToCoeffs(
              (lo.castShape(
                F_SPECIES,
                1
              ) as FloatVector?)!!
            )
            val hi0: FloatVector = mxfp4CodesToCoeffs(
              (hi.castShape(
                F_SPECIES,
                0
              ) as FloatVector?)!!
            )
            val hi1: FloatVector = mxfp4CodesToCoeffs(
              (hi.castShape(
                F_SPECIES,
                1
              ) as FloatVector?)!!
            )
            val x0 = that.getFloatVector(F_SPECIES, thatOffset + j)
            val x1 = that.getFloatVector(
              F_SPECIES,
              thatOffset + j + F_SPECIES.length()
            )
            val x2 =
              that.getFloatVector(F_SPECIES, thatOffset + j + QK_MXFP4 / 2)
            val x3 = that.getFloatVector(
              F_SPECIES,
              thatOffset + j + QK_MXFP4 / 2 + F_SPECIES.length()
            )
            blockSum += lo0.fma(x0, lo1.mul(x1)).reduceLanes(VectorOperators.ADD)
            blockSum += hi0.fma(x2, hi1.mul(x3)).reduceLanes(VectorOperators.ADD)
          }

          128 -> {
            var sum = FloatVector.zero(F_SPECIES)
            for (p in 0..3) {
              val loPart: FloatVector = mxfp4CodesToCoeffs(
                (lo.castShape(
                  F_SPECIES,
                  p
                ) as FloatVector?)!!
              )
              val hiPart: FloatVector = mxfp4CodesToCoeffs(
                (hi.castShape(
                  F_SPECIES,
                  p
                ) as FloatVector?)!!
              )
              val xLo = that.getFloatVector(
                F_SPECIES,
                thatOffset + j + p * F_SPECIES.length()
              )
              val xHi = that.getFloatVector(
                F_SPECIES,
                thatOffset + j + QK_MXFP4 / 2 + p * F_SPECIES.length()
              )
              sum = loPart.fma(xLo, sum)
              sum = hiPart.fma(xHi, sum)
            }
            blockSum += sum.reduceLanes(VectorOperators.ADD)
          }

          else -> throw UnsupportedOperationException(F_SPECIES.toString())
        }

        result += blockSum * d
        j += QK_MXFP4
      }

      if (j < size) {
        result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
      }
      return result
    }

    private fun mxfp4CodesToCoeffs(codes: FloatVector): FloatVector {
      val zero = FloatVector.zero(F_SPECIES)
      val eight = FloatVector.broadcast(F_SPECIES, 8f)
      val negMask = codes.compare(VectorOperators.GE, 8f)

      val t = codes.sub(zero.blend(eight, negMask))
      val mag = t
        .add(t.sub(4f).lanewise(VectorOperators.MAX, 0f))
        .add(t.sub(6f).lanewise(VectorOperators.MAX, 0f).mul(2f))
      return mag.blend(mag.neg(), negMask)
    }

    private fun scalarDot(
      thiz: MXFP4FloatTensor,
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

    private fun e8m0ToFp32Half(x: Int): Float {
      val bits: Int = if (x < 2) {
        0x00200000 shl x
      } else {
        (x - 1) shl 23
      }
      return Float.fromBits(bits)
    }
  }
}
