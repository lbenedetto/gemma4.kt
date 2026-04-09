package com.llama4j.floattensor

import com.llama4j.gguf.GGMLType
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.Byte
import java.lang.Float
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder
import java.util.*
import kotlin.Int
import kotlin.Long
import kotlin.UnsupportedOperationException
import kotlin.assert
import kotlin.collections.plus
import kotlin.intArrayOf
import kotlin.math.min
import kotlin.plus
import kotlin.sequences.plus
import kotlin.text.plus
import kotlin.times
import kotlin.toString

internal class MXFP4FloatTensor(size: Long, memorySegment: MemorySegment) : FloatTensor() {
  private val size: Long
  private val memorySegment: MemorySegment

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
    return GGMLType.MXFP4
  }

  override fun getFloat(index: Long): Float {
    assert(0 <= index && index < size)
    val blockIndex: Long = index / GGMLType.Companion.QK_MXFP4
    val inBlockIndex = (index % GGMLType.Companion.QK_MXFP4).toInt()
    val blockOffset = blockIndex * GGMLType.MXFP4.getTypeSize()

    val e8m0 = Byte.toUnsignedInt(FloatTensor.Companion.readByte(memorySegment, blockOffset))
    val d: Float = e8m0ToFp32Half(e8m0)

    val qsOffset = blockOffset + Byte.BYTES + (inBlockIndex and 0x0F)
    val packed = Byte.toUnsignedInt(FloatTensor.Companion.readByte(memorySegment, qsOffset))
    val nibble = if (inBlockIndex < (GGMLType.Companion.QK_MXFP4 / 2)) (packed and 0x0F) else ((packed ushr 4) and 0x0F)

    return MXFP4_VALUES[nibble] * d
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor) {
      if (FloatTensor.Companion.USE_VECTOR_API) {
        return vectorDot(this, thisOffset, that, thatOffset, size)
      }
      return scalarDot(this, thisOffset, that, thatOffset, size)
    }
    return FloatTensor.Companion.scalarDot(this, thisOffset, that, thatOffset, size)
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
      assert(Integer.bitCount(GGMLType.Companion.QK_MXFP4) == 1) { "power of 2" }
      var j = 0
      var result = 0f

      val alignmentBound = min(size, -thisOffset and (GGMLType.Companion.QK_MXFP4 - 1))
      if (alignmentBound > 0) {
        result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
        j = alignmentBound
      }

      val upperBound: Int = j + (size - j) / GGMLType.Companion.QK_MXFP4 * GGMLType.Companion.QK_MXFP4
      while (j < upperBound) {
        assert((thisOffset + j) % GGMLType.Companion.QK_MXFP4 == 0)
        val blockOffset: Long = (thisOffset + j).toLong() / GGMLType.Companion.QK_MXFP4 * GGMLType.MXFP4.getTypeSize()
        val d: Float =
          e8m0ToFp32Half(Byte.toUnsignedInt(FloatTensor.Companion.readByte(thiz.memorySegment, blockOffset)))

        val packed = ByteVector.fromMemorySegment(
          ByteVector.SPECIES_128,
          thiz.memorySegment,
          blockOffset + Byte.BYTES,
          ByteOrder.LITTLE_ENDIAN
        )
        val lo = packed.and(0x0F.toByte())
        val hi: ByteVector = packed.lanewise(VectorOperators.LSHR, 4)

        var blockSum = 0f
        when (Objects.requireNonNull<VectorSpecies<Float>?>(FloatTensor.Companion.F_SPECIES).vectorBitSize()) {
          512 -> {
            val loCoeffs: FloatVector = Companion.mxfp4CodesToCoeffs(
              (lo.castShape<kotlin.Float>(
                FloatTensor.Companion.F_SPECIES,
                0
              ) as FloatVector?)!!
            )
            val hiCoeffs: FloatVector = Companion.mxfp4CodesToCoeffs(
              (hi.castShape<kotlin.Float>(
                FloatTensor.Companion.F_SPECIES,
                0
              ) as FloatVector?)!!
            )
            val xLo = that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j)
            val xHi =
              that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j + GGMLType.Companion.QK_MXFP4 / 2)
            blockSum += loCoeffs.fma(xLo, hiCoeffs.mul(xHi)).reduceLanes(VectorOperators.ADD)
          }

          256 -> {
            val lo0: FloatVector = Companion.mxfp4CodesToCoeffs(
              (lo.castShape<kotlin.Float>(
                FloatTensor.Companion.F_SPECIES,
                0
              ) as FloatVector?)!!
            )
            val lo1: FloatVector = Companion.mxfp4CodesToCoeffs(
              (lo.castShape<kotlin.Float>(
                FloatTensor.Companion.F_SPECIES,
                1
              ) as FloatVector?)!!
            )
            val hi0: FloatVector = Companion.mxfp4CodesToCoeffs(
              (hi.castShape<kotlin.Float>(
                FloatTensor.Companion.F_SPECIES,
                0
              ) as FloatVector?)!!
            )
            val hi1: FloatVector = Companion.mxfp4CodesToCoeffs(
              (hi.castShape<kotlin.Float>(
                FloatTensor.Companion.F_SPECIES,
                1
              ) as FloatVector?)!!
            )
            val x0 = that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j)
            val x1 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + FloatTensor.Companion.F_SPECIES.length()
            )
            val x2 =
              that.getFloatVector(FloatTensor.Companion.F_SPECIES, thatOffset + j + GGMLType.Companion.QK_MXFP4 / 2)
            val x3 = that.getFloatVector(
              FloatTensor.Companion.F_SPECIES,
              thatOffset + j + GGMLType.Companion.QK_MXFP4 / 2 + FloatTensor.Companion.F_SPECIES.length()
            )
            blockSum += lo0.fma(x0, lo1.mul(x1)).reduceLanes(VectorOperators.ADD)
            blockSum += hi0.fma(x2, hi1.mul(x3)).reduceLanes(VectorOperators.ADD)
          }

          128 -> {
            var sum = FloatVector.zero(FloatTensor.Companion.F_SPECIES)
            for (p in 0..3) {
              val loPart: FloatVector = Companion.mxfp4CodesToCoeffs(
                (lo.castShape<kotlin.Float>(
                  FloatTensor.Companion.F_SPECIES,
                  p
                ) as FloatVector?)!!
              )
              val hiPart: FloatVector = Companion.mxfp4CodesToCoeffs(
                (hi.castShape<kotlin.Float>(
                  FloatTensor.Companion.F_SPECIES,
                  p
                ) as FloatVector?)!!
              )
              val xLo = that.getFloatVector(
                FloatTensor.Companion.F_SPECIES,
                thatOffset + j + p * FloatTensor.Companion.F_SPECIES.length()
              )
              val xHi = that.getFloatVector(
                FloatTensor.Companion.F_SPECIES,
                thatOffset + j + GGMLType.Companion.QK_MXFP4 / 2 + p * FloatTensor.Companion.F_SPECIES.length()
              )
              sum = loPart.fma(xLo, sum)
              sum = hiPart.fma(xHi, sum)
            }
            blockSum += sum.reduceLanes(VectorOperators.ADD)
          }

          else -> throw UnsupportedOperationException(FloatTensor.Companion.F_SPECIES.toString())
        }

        result += blockSum * d
        j += GGMLType.Companion.QK_MXFP4
      }

      if (j < size) {
        result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
      }
      return result
    }

    private fun mxfp4CodesToCoeffs(codes: FloatVector): FloatVector {
      val zero = FloatVector.zero(FloatTensor.Companion.F_SPECIES)
      val eight = FloatVector.broadcast(FloatTensor.Companion.F_SPECIES, 8f)
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
      val bits: Int
      if (x < 2) {
        bits = 0x00200000 shl x
      } else {
        bits = (x - 1) shl 23
      }
      return Float.intBitsToFloat(bits)
    }
  }
}
