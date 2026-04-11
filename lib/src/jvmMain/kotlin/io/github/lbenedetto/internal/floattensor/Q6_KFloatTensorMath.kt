package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.Q6_KFloatTensor.Companion.BLOCK_SIZE
import io.github.lbenedetto.internal.floattensor.Q6_KFloatTensor.Companion.TYPE_SIZE
import io.github.lbenedetto.internal.floattensor.Q6_KFloatTensor.Companion.fp16ToFloatNoIntrinsic
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.F_SPECIES
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import java.nio.ByteOrder
import kotlin.math.min

actual object Q6_KFloatTensorMath {
  internal actual fun vectorDot(
    thiz: Q6_KFloatTensor,
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

    var acc = FloatVector.zero(F_SPECIES!!)
    var blockOffset: Long = (thisOffset + j).toLong() / BLOCK_SIZE * TYPE_SIZE
    val upperBound: Int = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE

    while (j < upperBound) {
      val qlOff = blockOffset
      val qhOff = blockOffset + 128
      val scOff = blockOffset + 192
      // NOTE: Deliberately avoid Float.float16ToFloat here.
      // In native-image builds, Graal can lower that intrinsic to VCVTPH2PS with
      // an illegal high XMM operand under heavy vector register pressure in Q6_K
      // vectorDot, causing a compile-time crash. Keep this software conversion
      // until the Graal backend bug is fixed.
      val d: Float =
        fp16ToFloatNoIntrinsic(thiz.memorySegment.readShort(blockOffset + 208))

      for (h in 0..1) {
        val qlBase = qlOff + h * 64
        val qhBase = qhOff + h * 32

        val base = thatOffset + j + h * 128
        for (c in 0..1) {
          val qlA = ByteVector.fromMemorySegment(
            ByteVector.SPECIES_128, thiz.memorySegment.actual(),
            qlBase + c * 16L, ByteOrder.LITTLE_ENDIAN
          )
          val qlB = ByteVector.fromMemorySegment(
            ByteVector.SPECIES_128, thiz.memorySegment.actual(),
            qlBase + 32 + c * 16L, ByteOrder.LITTLE_ENDIAN
          )
          val qhV = ByteVector.fromMemorySegment(
            ByteVector.SPECIES_128, thiz.memorySegment.actual(),
            qhBase + c * 16L, ByteOrder.LITTLE_ENDIAN
          )

          val q0: ByteVector = qlA
            .and(0xF.toByte())
            .or(qhV.and(3.toByte()).lanewise(VectorOperators.LSHL, 4L))
            .sub(32.toByte())
          val q1: ByteVector = qlB
            .and(0xF.toByte())
            .or(qhV.lanewise(VectorOperators.LSHR, 2L).and(3.toByte()).lanewise(VectorOperators.LSHL, 4L))
            .sub(32.toByte())
          val q2: ByteVector = qlA
            .lanewise(VectorOperators.LSHR, 4L)
            .or(qhV.lanewise(VectorOperators.LSHR, 4L).and(3.toByte()).lanewise(VectorOperators.LSHL, 4L))
            .sub(32.toByte())
          val q3: ByteVector = qlB
            .lanewise(VectorOperators.LSHR, 4L)
            .or(qhV.lanewise(VectorOperators.LSHR, 6L).and(3.toByte()).lanewise(VectorOperators.LSHL, 4L))
            .sub(32.toByte())

          val ds0: Float = d * thiz.memorySegment.readByte(scOff + h * 8 + c)
          val ds1: Float = d * thiz.memorySegment.readByte(scOff + h * 8 + 2 + c)
          val ds2: Float = d * thiz.memorySegment.readByte(scOff + h * 8 + 4 + c)
          val ds3: Float = d * thiz.memorySegment.readByte(scOff + h * 8 + 6 + c)

          val ds0Vec = FloatVector.broadcast(F_SPECIES, ds0)
          val ds1Vec = FloatVector.broadcast(F_SPECIES, ds1)
          val ds2Vec = FloatVector.broadcast(F_SPECIES, ds2)
          val ds3Vec = FloatVector.broadcast(F_SPECIES, ds3)

          val sg0Idx = base + c * 16
          val sg1Idx = base + 32 + c * 16
          val sg2Idx = base + 64 + c * 16
          val sg3Idx = base + 96 + c * 16

          when (F_SPECIES.vectorBitSize()) {
            512 -> {
              val q0f = q0.castShape(F_SPECIES, 0).reinterpretAsFloats()
              val q1f = q1.castShape(F_SPECIES, 0).reinterpretAsFloats()
              val q2f = q2.castShape(F_SPECIES, 0).reinterpretAsFloats()
              val q3f = q3.castShape(F_SPECIES, 0).reinterpretAsFloats()
              acc = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx), acc)
              acc = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx), acc)
              acc = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx), acc)
              acc = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx), acc)
            }

            256 -> {
              for (p in 0..1) {
                val off: Int = p * F_SPECIES.length()
                val q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats()
                val q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats()
                val q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats()
                val q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats()
                acc = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), acc)
                acc = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), acc)
                acc = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), acc)
                acc = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), acc)
              }
            }

            128 -> {
              for (p in 0..3) {
                val off: Int = p * F_SPECIES.length()
                val q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats()
                val q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats()
                val q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats()
                val q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats()
                acc = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), acc)
                acc = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), acc)
                acc = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), acc)
                acc = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), acc)
              }
            }

            else -> throw UnsupportedOperationException(F_SPECIES.toString())
          }
        }
      }
      j += BLOCK_SIZE
      blockOffset += TYPE_SIZE.toLong()
    }

    result += acc.reduceLanes(VectorOperators.ADD)

    if (j < size) {
      result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
    }

    return result
  }

}
