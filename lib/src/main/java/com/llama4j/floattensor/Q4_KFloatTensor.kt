package com.llama4j.floattensor

import com.llama4j.gguf.GGMLType
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.Byte
import java.lang.foreign.MemorySegment
import java.nio.ByteOrder
import java.util.*
import kotlin.Boolean
import kotlin.Float
import kotlin.Int
import kotlin.Long
import kotlin.UnsupportedOperationException
import kotlin.assert
import kotlin.math.min
import kotlin.toString

internal class Q4_KFloatTensor(size: Long, memorySegment: MemorySegment) : FloatTensor() {
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
    return GGMLType.Q4_K
  }

  override fun getFloat(index: Long): Float {
    val blockIndex: Long = index / BLOCK_SIZE
    val withinBlock = (index % BLOCK_SIZE).toInt()
    val blockOffset: Long = blockIndex * TYPE_SIZE
    val d: Float = FloatTensor.Companion.readFloat16(memorySegment, blockOffset)
    val dmin: Float = FloatTensor.Companion.readFloat16(memorySegment, blockOffset + 2)
    val scalesOffset = blockOffset + 4
    val qsOffset = blockOffset + 16 // 4 + 12

    // Each group of 64 values uses 2 sub-blocks: low nibble (32) + high nibble (32)
    val group = withinBlock / 64 // 0..3
    val inGroup = withinBlock % 64
    val subBlock: Int
    val nibbleIndex: Int
    val isHigh: Boolean
    if (inGroup < 32) {
      subBlock = group * 2
      nibbleIndex = inGroup
      isHigh = false
    } else {
      subBlock = group * 2 + 1
      nibbleIndex = inGroup - 32
      isHigh = true
    }

    val sc: Int = getScaleMinK4(subBlock, memorySegment, scalesOffset, false)
    val m: Int = getScaleMinK4(subBlock, memorySegment, scalesOffset, true)

    val qsByte: Byte = FloatTensor.Companion.readByte(memorySegment, qsOffset + group * 32 + nibbleIndex)
    val quant = if (isHigh) ((Byte.toUnsignedInt(qsByte) shr 4) and 0xF) else (Byte.toUnsignedInt(qsByte) and 0xF)

    return d * sc * quant - dmin * m
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (FloatTensor.Companion.USE_VECTOR_API) {
      return vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      return FloatTensor.Companion.scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }

  companion object {
    val BLOCK_SIZE: Int = GGMLType.Companion.QK_K
    val TYPE_SIZE: Int = GGMLType.Q4_K.getTypeSize()

    // Decode scale or min for sub-block j (0..7) from the 12-byte scales array
    fun getScaleMinK4(j: Int, mem: MemorySegment, scalesOffset: Long, isMin: Boolean): Int {
      if (j < 4) {
        val idx = if (isMin) j + 4 else j
        return Byte.toUnsignedInt(FloatTensor.Companion.readByte(mem, scalesOffset + idx)) and 63
      } else {
        val lowIdx = j + 4
        val highIdx = if (isMin) j else j - 4
        val low = if (isMin)
          (Byte.toUnsignedInt(FloatTensor.Companion.readByte(mem, scalesOffset + lowIdx)) shr 4)
        else
          (Byte.toUnsignedInt(FloatTensor.Companion.readByte(mem, scalesOffset + lowIdx)) and 0xF)
        val high = (Byte.toUnsignedInt(FloatTensor.Companion.readByte(mem, scalesOffset + highIdx)) shr 6) and 0x3
        return low or (high shl 4)
      }
    }

    private fun vectorDot(
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
        result += FloatTensor.Companion.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
        j += alignmentBound
      }

      var `val` = FloatVector.zero(Objects.requireNonNull<VectorSpecies<Float>?>(FloatTensor.Companion.F_SPECIES))
      var val2 = FloatVector.zero(FloatTensor.Companion.F_SPECIES)
      var blockOffset: Long = (thisOffset + j).toLong() / BLOCK_SIZE * TYPE_SIZE
      val upperBound: Int = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE

      while (j < upperBound) {
        val d: Float = FloatTensor.Companion.readFloat16(thiz.memorySegment, blockOffset)
        val dmin: Float = FloatTensor.Companion.readFloat16(thiz.memorySegment, blockOffset + 2)
        val scalesOff = blockOffset + 4
        val qsOff = blockOffset + 16

        // 4 groups of 64 values each (2 sub-blocks per group: low nibble + high nibble)
        for (g in 0..3) {
          val d1: Float = d * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, false)
          val negM1: Float = -(dmin * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, true))
          val d2: Float = d * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, false)
          val negM2: Float = -(dmin * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, true))

          val d1Vec = FloatVector.broadcast(FloatTensor.Companion.F_SPECIES, d1)
          val negM1Vec = FloatVector.broadcast(FloatTensor.Companion.F_SPECIES, negM1)
          val d2Vec = FloatVector.broadcast(FloatTensor.Companion.F_SPECIES, d2)
          val negM2Vec = FloatVector.broadcast(FloatTensor.Companion.F_SPECIES, negM2)

          val loBase = thatOffset + j + g * 64
          val hiBase = thatOffset + j + g * 64 + 32

          // Process 32 bytes of qs in 2 chunks of 16 bytes
          for (c in 0..1) {
            val wBytes = ByteVector.fromMemorySegment(
              ByteVector.SPECIES_128, thiz.memorySegment,
              qsOff + g.toLong() * 32 + c * 16, ByteOrder.LITTLE_ENDIAN
            )
            val loBytes = wBytes.and(0xF.toByte())
            val hiBytes: ByteVector = wBytes.lanewise(VectorOperators.LSHR, 4)

            val loIdx = loBase + c * 16
            val hiIdx = hiBase + c * 16

            when (FloatTensor.Companion.F_SPECIES.vectorBitSize()) {
              512 -> {
                val loQ = loBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0).reinterpretAsFloats()
                `val` = loQ.fma(d1Vec, negM1Vec).fma(that.getFloatVector(FloatTensor.Companion.F_SPECIES, loIdx), `val`)
                val hiQ = hiBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0).reinterpretAsFloats()
                val2 = hiQ.fma(d2Vec, negM2Vec).fma(that.getFloatVector(FloatTensor.Companion.F_SPECIES, hiIdx), val2)
              }

              256 -> {
                val loQ0 = loBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0).reinterpretAsFloats()
                val loQ1 = loBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 1).reinterpretAsFloats()
                `val` =
                  loQ0.fma(d1Vec, negM1Vec).fma(that.getFloatVector(FloatTensor.Companion.F_SPECIES, loIdx), `val`)
                val2 = loQ1.fma(d1Vec, negM1Vec).fma(
                  that.getFloatVector(
                    FloatTensor.Companion.F_SPECIES,
                    loIdx + FloatTensor.Companion.F_SPECIES.length()
                  ), val2
                )
                val hiQ0 = hiBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 0).reinterpretAsFloats()
                val hiQ1 = hiBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, 1).reinterpretAsFloats()
                `val` =
                  hiQ0.fma(d2Vec, negM2Vec).fma(that.getFloatVector(FloatTensor.Companion.F_SPECIES, hiIdx), `val`)
                val2 = hiQ1.fma(d2Vec, negM2Vec).fma(
                  that.getFloatVector(
                    FloatTensor.Companion.F_SPECIES,
                    hiIdx + FloatTensor.Companion.F_SPECIES.length()
                  ), val2
                )
              }

              128 -> {
                for (p in 0..3) {
                  val loQ = loBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, p).reinterpretAsFloats()
                  `val` = loQ.fma(d1Vec, negM1Vec).fma(
                    that.getFloatVector(
                      FloatTensor.Companion.F_SPECIES,
                      loIdx + p * FloatTensor.Companion.F_SPECIES.length()
                    ), `val`
                  )
                  val hiQ = hiBytes.castShape<Float>(FloatTensor.Companion.F_SPECIES, p).reinterpretAsFloats()
                  val2 = hiQ.fma(d2Vec, negM2Vec).fma(
                    that.getFloatVector(
                      FloatTensor.Companion.F_SPECIES,
                      hiIdx + p * FloatTensor.Companion.F_SPECIES.length()
                    ), val2
                  )
                }
              }

              else -> throw UnsupportedOperationException(FloatTensor.Companion.F_SPECIES.toString())
            }
          }
        }
        j += BLOCK_SIZE
        blockOffset += TYPE_SIZE.toLong()
      }
      result += `val`.add(val2).reduceLanes(VectorOperators.ADD)

      // Handle tail
      if (j < size) {
        result += FloatTensor.Companion.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
      }

      return result
    }
  }
}
