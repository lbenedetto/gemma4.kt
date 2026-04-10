package io.github.lbenedetto.internal.floattensor

import io.github.lbenedetto.internal.gguf.GGMLType
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorShape
import jdk.incubator.vector.VectorSpecies
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout

internal abstract class AbstractFloatTensor : FloatTensor {
  abstract fun getFloatVector(species: VectorSpecies<Float>, offset: Int): FloatVector?

  abstract fun type(): GGMLType?

  companion object {
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
  }
}
