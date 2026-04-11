package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.toUnsignedInt
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.util.MemorySegment
import io.github.lbenedetto.internal.util.vectorMathEnabled
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies

internal class Q4_1FloatTensor(
  override val size: Long,
  val memorySegment: MemorySegment
) : AbstractFloatTensor() {

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("setFloat")
  }

  override fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector? {
    throw UnsupportedOperationException("getFloatVector")
  }

  override fun type(): GGMLType {
    return GGMLType.Q4_1
  }

  override fun getFloat(index: Long): Float {
    assert(index in 0..<size)
    val blockIndex = index / GGMLType.Q4_1.blockSize
    val blockOffset = blockIndex * GGMLType.Q4_1.typeSize
    val delta: Float = memorySegment.readFloat16(blockOffset)
    val min: Float = memorySegment.readFloat16(blockOffset + Float16.BYTES)
    val modIndex = (index % GGMLType.Q4_1.blockSize).toInt()
    val quant: Int = if (modIndex < 16) {
      val offset = blockOffset + 2 * Float16.BYTES + modIndex
      memorySegment.readByte(offset).toUnsignedInt() and 0x0F
    } else {
      val offset = blockOffset + 2 * Float16.BYTES + modIndex - 16
      (memorySegment.readByte(offset).toUnsignedInt() ushr 4) and 0x0F
    }
    return delta * quant + min
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (vectorMathEnabled()) {
      Q4_1FloatTensorMath.vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }
}
