package io.github.lbenedetto.internal.util

expect class MemorySegment {
  fun readShort(offset: Long): Short
  fun readFloat16(offset: Long): Float
  fun readByte(offset: Long): Byte
  fun readFloat(offset: Long): Float
  fun asFloatBuffer(): FloatBuffer
}
