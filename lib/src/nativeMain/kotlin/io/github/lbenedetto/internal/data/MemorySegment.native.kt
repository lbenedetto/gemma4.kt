package io.github.lbenedetto.internal.data

import kotlinx.cinterop.*
import okio.Path
import platform.posix.*

@OptIn(ExperimentalForeignApi::class)
actual class MemorySegment(private val ptr: CPointer<ByteVar>, private val size: Long) {

  actual fun readByte(offset: Long): Byte = ptr[offset.toInt()]

  actual fun readShort(offset: Long): Short {
    val b0 = ptr[offset.toInt()].toInt() and 0xFF
    val b1 = ptr[(offset + 1).toInt()].toInt() and 0xFF
    return ((b1 shl 8) or b0).toShort()
  }

  actual fun readFloat16(offset: Long): Float = float16ToFloat(readShort(offset))

  actual fun readFloat(offset: Long): Float {
    val b0 = ptr[offset.toInt()].toInt() and 0xFF
    val b1 = ptr[(offset + 1).toInt()].toInt() and 0xFF
    val b2 = ptr[(offset + 2).toInt()].toInt() and 0xFF
    val b3 = ptr[(offset + 3).toInt()].toInt() and 0xFF
    return Float.fromBits((b3 shl 24) or (b2 shl 16) or (b1 shl 8) or b0)
  }

  actual fun slice(offset: Long, size: Long): MemorySegment =
    MemorySegment((ptr + offset)!!, size)

  actual fun asFloatBuffer(): FloatBuffer {
    val count = (size / 4).toInt()
    val arr = FloatArray(count) { i ->
      val base = i.toLong() * 4
      val b0 = ptr[base.toInt()].toInt() and 0xFF
      val b1 = ptr[(base + 1).toInt()].toInt() and 0xFF
      val b2 = ptr[(base + 2).toInt()].toInt() and 0xFF
      val b3 = ptr[(base + 3).toInt()].toInt() and 0xFF
      Float.fromBits((b3 shl 24) or (b2 shl 16) or (b1 shl 8) or b0)
    }
    return FloatBuffer(arr)
  }

  actual companion object {
    actual fun mmap(path: Path, offset: Long, size: Long): MemorySegment {
      val fd = open(path.toString(), O_RDONLY)
      require(fd >= 0) { "Failed to open $path" }
      try {
        val ptr = mmap(null, size.toULong(), PROT_READ, MAP_PRIVATE, fd, offset)
        require(ptr != MAP_FAILED) { "mmap failed for $path at offset $offset size $size" }
        return MemorySegment(ptr!!.reinterpret(), size)
      } finally {
        close(fd)
      }
    }
  }
}

private fun float16ToFloat(half: Short): Float {
  val h = half.toInt() and 0xFFFF
  val sign = (h ushr 15) shl 31
  val exp = (h ushr 10) and 0x1F
  val frac = h and 0x3FF
  val bits = when {
    exp == 0 && frac == 0 -> sign
    exp == 0x1F -> sign or (0xFF shl 23) or (frac shl 13)
    else -> sign or ((exp + 112) shl 23) or (frac shl 13)
  }
  return Float.fromBits(bits)
}
