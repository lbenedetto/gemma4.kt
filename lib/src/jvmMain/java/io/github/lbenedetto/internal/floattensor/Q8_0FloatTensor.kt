package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.util.MemorySegment
import io.github.lbenedetto.internal.util.vectorMathEnabled
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies

internal class Q8_0FloatTensor(
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
    return GGMLType.Q8_0
  }

  override fun getFloat(index: Long): Float {
    val blockIndex = index / GGMLType.Q8_0.blockSize
    val withinBlockIndex = index % GGMLType.Q8_0.blockSize
    val blockOffset = blockIndex * GGMLType.Q8_0.typeSize
    val quant: Byte = memorySegment.readByte(blockOffset + Float16.BYTES + withinBlockIndex)
    val scale: Float = memorySegment.readFloat16(blockOffset)
    return quant * scale
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (vectorMathEnabled()) {
      Q8_0FloatTensorMath.vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }
}
