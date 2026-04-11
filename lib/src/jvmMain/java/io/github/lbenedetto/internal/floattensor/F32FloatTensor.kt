package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.floattensor.VectorSpeciesConfig.USE_VECTOR_API
import io.github.lbenedetto.internal.gguf.GGMLType
import io.github.lbenedetto.internal.util.MemorySegment
import io.github.lbenedetto.internal.util.vectorMathEnabled
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies
import java.nio.ByteOrder

internal class F32FloatTensor(
  override val size: Long,
  internal val memorySegment: MemorySegment
) : AbstractFloatTensor() {

  override fun getFloat(index: Long): Float {
    return memorySegment.readFloat(index * Float.SIZE_BYTES)
  }

  override fun setFloat(index: Int, value: Float) {
    throw UnsupportedOperationException("read-only")
  }

  override fun type(): GGMLType {
    return GGMLType.F32
  }

  override fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector {
    if (!vectorMathEnabled()) {
      throw UnsupportedOperationException()
    }
    return FloatVector.fromMemorySegment(
      species,
      memorySegment.actual(),
      offset.toLong() * Float.SIZE_BYTES,
      ByteOrder.LITTLE_ENDIAN
    )
  }

  override fun dot(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    if (that is ArrayFloatTensor && USE_VECTOR_API) {
      return F32FloatTensorMath.vectorDot(this, thisOffset, that, thatOffset, size)
    }
    return scalarDot(this, thisOffset, that, thatOffset, size)
  }
}
