package io.github.lbenedetto.internal.data

import okio.Path

expect class MemorySegment {
  fun readShort(offset: Long): Short
  fun readFloat16(offset: Long): Float
  fun readByte(offset: Long): Byte
  fun readFloat(offset: Long): Float
  fun asFloatBuffer(): FloatBuffer
  fun slice(offset: Long, size: Long): MemorySegment

  companion object {
    fun mmap(path: Path, offset: Long, size: Long): MemorySegment
  }
}
