package io.github.lbenedetto.internal.data

import kotlinx.cinterop.*
import okio.Path
import platform.posix.*

@OptIn(ExperimentalForeignApi::class)
actual class MemorySegment(private val ptr: CPointer<ByteVar>, private val size: Long) {

  /** Returns the raw pointer offset by the given number of bytes, for use with C interop. */
  internal fun rawPointer(byteOffset: Long = 0): CPointer<ByteVar> = (ptr + byteOffset)!!

  actual fun readByte(offset: Long): Byte = (ptr + offset)!![0]

  actual fun readShort(offset: Long): Short {
    val p = (ptr + offset)!!
    val b0 = p[0].toInt() and 0xFF
    val b1 = p[1].toInt() and 0xFF
    return ((b1 shl 8) or b0).toShort()
  }

  actual fun readFloat16(offset: Long): Float = float16ToFloat(readShort(offset))

  actual fun readFloat(offset: Long): Float {
    val p = (ptr + offset)!!
    val b0 = p[0].toInt() and 0xFF
    val b1 = p[1].toInt() and 0xFF
    val b2 = p[2].toInt() and 0xFF
    val b3 = p[3].toInt() and 0xFF
    return Float.fromBits((b3 shl 24) or (b2 shl 16) or (b1 shl 8) or b0)
  }

  actual fun slice(offset: Long, size: Long): MemorySegment =
    MemorySegment((ptr + offset)!!, size)

  actual fun asFloatBuffer(): FloatBuffer {
    val count = (size / 4).toInt()
    val arr = FloatArray(count) { i ->
      readFloat(i.toLong() * 4)
    }
    return FloatBuffer(arr)
  }

  actual companion object {
    actual fun mmap(path: Path, offset: Long, size: Long): MemorySegment {
      val fd = open(path.toString(), O_RDONLY)
      require(fd >= 0) { "Failed to open $path" }
      try {
        val pageSize = sysconf(_SC_PAGESIZE)
        val alignedOffset = offset / pageSize * pageSize
        val delta = offset - alignedOffset
        val mappedSize = size + delta
        val ptr = mmap(null, mappedSize.toULong(), PROT_READ, MAP_PRIVATE, fd, alignedOffset)
        require(ptr != MAP_FAILED) { "mmap failed for $path at offset $offset size $size" }
        return MemorySegment((ptr!!.reinterpret<ByteVar>() + delta)!!, size)
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
