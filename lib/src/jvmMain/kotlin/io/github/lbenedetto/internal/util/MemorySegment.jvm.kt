package io.github.lbenedetto.internal.util

import okio.Path
import java.lang.foreign.Arena
import java.lang.foreign.ValueLayout
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
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

  actual fun slice(offset: Long, size: Long): MemorySegment {
    return MemorySegment(memorySegment.asSlice(offset, size))
  }

  fun actual() = memorySegment

  actual companion object {
    actual fun mmap(path: Path, offset: Long, size: Long): MemorySegment {
      val jpath = java.nio.file.Path.of(path.toString())
      FileChannel.open(jpath, StandardOpenOption.READ).use { fc ->
        return MemorySegment(fc.map(FileChannel.MapMode.READ_ONLY, offset, size, Arena.global()))
      }
    }
  }
}
