package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.util.MemorySegment
import io.github.lbenedetto.internal.util.vectorMathEnabled
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies

internal class BF16FloatTensor(
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
    return GGMLType.BF16
  }

  override fun getFloat(index: Long): Float {
    assert(index in 0..<size)
    val bits: Short = memorySegment.readShort(index * 2)
    return Float.fromBits(bits.toInt() shl 16)
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    return if (vectorMathEnabled()) {
      BF16FloatTensorMath.vectorDot(this, thisOffset, that as ArrayFloatTensor, thatOffset, size)
    } else {
      scalarDot(this, thisOffset, that, thatOffset, size)
    }
  }
}
