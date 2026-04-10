package io.github.lbenedetto.internal.util

import java.lang.foreign.ValueLayout
import java.nio.ByteOrder
import java.lang.foreign.MemorySegment as JMemorySegment

actual class MemorySegment(private val memorySegment: JMemorySegment) {
  actual fun readShort(offset: Long): Short {
    return memorySegment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset)
  }

  actual fun readFloat16(offset: Long): Float {
    return java.lang.Float.float16ToFloat(readShort(offset))
  }

  actual fun readByte(offset: Long): Byte {
    return memorySegment.get(ValueLayout.JAVA_BYTE, offset)
  }

  actual fun readFloat(offset: Long): Float {
    return memorySegment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset)
  }

  actual fun asFloatBuffer(): FloatBuffer {
    return memorySegment.asByteBuffer()
      .order(ByteOrder.LITTLE_ENDIAN)
      .asFloatBuffer()
      .let { FloatBuffer(it) }
  }

  fun actual() = memorySegment
}
