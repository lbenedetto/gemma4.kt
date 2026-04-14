package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.gguf.GGMLType
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder
import kotlin.math.min

internal class Q4_0FloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : FloatTensor() {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector? {
    throw UnsupportedOperationException("getFloatVector")
  }

  override fun getFloat(index: Long): Float {
    assert(index in 0..<size)
    val blockIndex = index / GGMLType.Q4_0.blockSize
    val blockOffset = blockIndex * GGMLType.Q4_0.typeSize
    val scale: Float = readFloat16(memorySegment, blockOffset)
    var quant: Byte
    val modIndex = (index % GGMLType.Q4_0.blockSize).toInt()
    quant = if (modIndex < GGMLType.Q4_0.blockSize / 2) {
      (readByte(memorySegment, blockOffset + Float16.BYTES + modIndex)
        .toInt() and 0x0F).toByte()
    } else {
      ((readByte(
        memorySegment,
        blockOffset + Float16.BYTES + modIndex - GGMLType.Q4_0.blockSize / 2
      ).toInt() ushr 4) and 0x0F).toByte()
    }
    quant = (quant - 8).toByte()
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
      thiz: Q4_0FloatTensor,
      thisOffset: Int,
      that: ArrayFloatTensor,
      thatOffset: Int,
      size: Int
    ): Float {
      var result = 0f
      var j = 0

      assert(Integer.bitCount(GGMLType.Q4_0.blockSize) == 1) { "power of 2" }
      val alignmentBound = min(size, -thisOffset and (GGMLType.Q4_0.blockSize - 1))
      if (alignmentBound > 0) {
        result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
        j += alignmentBound
      }
      assert((thisOffset + j) % GGMLType.Q4_0.blockSize == 0)

      var `val` = FloatVector.zero(F_SPECIES!!)
      var blockOffset = (thisOffset + j).toLong() / GGMLType.Q4_0.blockSize * GGMLType.Q4_0.typeSize
      val upperBound = j + (size - j) / GGMLType.Q4_0.blockSize * GGMLType.Q4_0.blockSize
      while (j < upperBound) {
        val wScaleValue: Float = readFloat16(thiz.memorySegment, blockOffset)
        val wScale = FloatVector.broadcast(F_SPECIES, wScaleValue)
        val wBytes = ByteVector.fromMemorySegment(
          ByteVector.SPECIES_128,
          thiz.memorySegment,
          blockOffset + Float16.BYTES,
          ByteOrder.LITTLE_ENDIAN
        )
        val loBytes = wBytes.and(0xF.toByte()).sub(8.toByte())
        val hiBytes: ByteVector = wBytes.lanewise(VectorOperators.LSHR, 4L).sub(8.toByte())
        when (F_SPECIES.vectorBitSize()) {
          512 -> {
            val s0 = that.getFloatVector(F_SPECIES, thatOffset + j)
              .mul(loBytes.castShape(F_SPECIES, 0))
            val s1 = that.getFloatVector(
              F_SPECIES,
              thatOffset + j + F_SPECIES.length()
            ).mul(hiBytes.castShape(F_SPECIES, 0))
            `val` = s0.add(s1).fma(wScale, `val`)
          }

          256 -> {
            var s0 = that.getFloatVector(F_SPECIES, thatOffset + j)
              .mul(loBytes.castShape(F_SPECIES, 0))
            var s1 = that.getFloatVector(
              F_SPECIES,
              thatOffset + j + 2 * F_SPECIES.length()
            ).mul(hiBytes.castShape(F_SPECIES, 0))
            s0 = that.getFloatVector(
              F_SPECIES,
              thatOffset + j + F_SPECIES.length()
            ).fma(loBytes.castShape(F_SPECIES, 1), s0)
            s1 = that.getFloatVector(
              F_SPECIES,
              thatOffset + j + 3 * F_SPECIES.length()
            ).fma(hiBytes.castShape(F_SPECIES, 1), s1)
            `val` = s0.add(s1).fma(wScale, `val`)
          }

          128 -> {
            for (i in 0..1) {
              val tmp = if (i == 0) loBytes else hiBytes
              var s0 = that.getFloatVector(
                F_SPECIES,
                thatOffset + j + (i * 4) * F_SPECIES.length()
              ).mul(tmp.castShape(F_SPECIES, 0))
              var s1 = that.getFloatVector(
                F_SPECIES,
                thatOffset + j + (i * 4 + 2) * F_SPECIES.length()
              ).mul(tmp.castShape(F_SPECIES, 2))
              s0 = that.getFloatVector(
                F_SPECIES,
                thatOffset + j + (i * 4 + 1) * F_SPECIES.length()
              ).fma(tmp.castShape(F_SPECIES, 1), s0)
              s1 = that.getFloatVector(
                F_SPECIES,
                thatOffset + j + (i * 4 + 3) * F_SPECIES.length()
              ).fma(tmp.castShape(F_SPECIES, 3), s1)
              `val` = s0.add(s1).fma(wScale, `val`)
            }
          }

          else -> throw UnsupportedOperationException(F_SPECIES.toString())
        }
        j += GGMLType.Q4_0.blockSize
        blockOffset += GGMLType.Q4_0.typeSize.toLong()
      }
      result += `val`.reduceLanes(VectorOperators.ADD)

      if (j < size) {
        result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
      }

      return result
    }
  }
}
