package io.github.lbenedetto.internal.floattensor

import jdk.incubator.vector.VectorShape
import jdk.incubator.vector.VectorSpecies
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout

internal object FloatTensorHelpers {
  val VECTOR_BIT_SIZE: Int = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize())
  val USE_VECTOR_API: Boolean = VECTOR_BIT_SIZE != 0

  val F_SPECIES: VectorSpecies<Float>?
  val I_SPECIES: VectorSpecies<Int>?
  val S_SPECIES_HALF: VectorSpecies<Short>?

  init {
    if (USE_VECTOR_API) {
      F_SPECIES =
        VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(Float::class.javaPrimitiveType)
      I_SPECIES = F_SPECIES.withLanes(Int::class.javaPrimitiveType)
      S_SPECIES_HALF =
        VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(Short::class.javaPrimitiveType)
      assert(F_SPECIES.length() == S_SPECIES_HALF.length())
    } else {
      F_SPECIES = null
      I_SPECIES = null
      S_SPECIES_HALF = null
    }
  }

  fun Byte.toUnsignedInt(): Int = toInt() and 0xFF

  fun Short.toUnsignedInt(): Int = toInt() and 0xFFFF

  fun readShort(memorySegment: MemorySegment, offset: Long): Short {
    return memorySegment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset)
  }

  fun readFloat16(memorySegment: MemorySegment, offset: Long): Float {
    return java.lang.Float.float16ToFloat(readShort(memorySegment, offset))
  }

  fun readByte(memorySegment: MemorySegment, offset: Long): Byte {
    return memorySegment.get(ValueLayout.JAVA_BYTE, offset)
  }

  fun readFloat(memorySegment: MemorySegment, offset: Long): Float {
    return memorySegment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset)
  }

  fun numberOfElements(vararg dimensions: Int): Int {
    assert(dimensions.all { it > 0 })
    return dimensions.reduce(Math::multiplyExact)
  }

  fun numberOfElementsLong(vararg dimensions: Int): Long {
    var result: Long = 1
    for (d in dimensions) {
      assert(d > 0)
      result = Math.multiplyExact(result, d)
    }
    return result
  }

  fun scalarDot(thiz: FloatTensor, thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): Float {
    var result = 0f
    for (j in 0..<size) {
      result += thiz[thisOffset + j] * that[thatOffset + j]
    }
    return result
  }
}
