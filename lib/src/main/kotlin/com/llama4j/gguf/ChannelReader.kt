package com.llama4j.gguf

import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.ReadableByteChannel
import kotlin.math.min

internal class ChannelReader(private val channel: ReadableByteChannel, bufferSize: Int) {
  private val buffer: ByteBuffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.LITTLE_ENDIAN)
  private var position: Long

  init {
    this.buffer.limit(0)
    this.position = 0L
  }

  fun position(): Long {
    return position
  }

  @Throws(IOException::class)
  private fun ensure(required: Int) {
    require(required <= buffer.capacity()) { "Requested read " + required + " exceeds buffer capacity " + buffer.capacity() }
    if (buffer.remaining() >= required) {
      return
    }
    buffer.compact()
    while (buffer.position() < required) {
      val read = channel.read(buffer)
      if (read < 0) {
        throw IOException("Unexpected EOF while reading GGUF metadata")
      }
    }
    buffer.flip()
  }

  @Throws(IOException::class)
  fun readByte(): Byte {
    ensure(Byte.SIZE_BYTES)
    position += Byte.SIZE_BYTES.toLong()
    return buffer.get()
  }

  @Throws(IOException::class)
  fun readShort(): Short {
    ensure(Short.SIZE_BYTES)
    position += Short.SIZE_BYTES.toLong()
    return buffer.getShort()
  }

  @Throws(IOException::class)
  fun readInt(): Int {
    ensure(Integer.BYTES)
    position += Integer.BYTES.toLong()
    return buffer.getInt()
  }

  @Throws(IOException::class)
  fun readLong(): Long {
    ensure(Long.SIZE_BYTES)
    position += Long.SIZE_BYTES.toLong()
    return buffer.getLong()
  }

  @Throws(IOException::class)
  fun readFloat(): Float {
    return Float.fromBits(readInt())
  }

  @Throws(IOException::class)
  fun readDouble(): Double {
    return Double.fromBits(readLong())
  }

  @Throws(IOException::class)
  fun readBytes(length: Int): ByteArray {
    val bytes = ByteArray(length)
    var copied = 0
    while (copied < length) {
      if (!buffer.hasRemaining()) {
        ensure(1)
      }
      val chunk = min(length - copied, buffer.remaining())
      buffer.get(bytes, copied, chunk)
      copied += chunk
      position += chunk.toLong()
    }
    return bytes
  }

  @Throws(IOException::class)
  fun skipBytes(length: Int) {
    var remaining = length
    while (remaining > 0) {
      if (!buffer.hasRemaining()) {
        ensure(1)
      }
      val chunk = min(remaining, buffer.remaining())
      buffer.position(buffer.position() + chunk)
      remaining -= chunk
      position += chunk.toLong()
    }
  }
}
