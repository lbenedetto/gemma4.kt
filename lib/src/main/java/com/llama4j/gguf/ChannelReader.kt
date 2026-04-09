package com.llama4j.gguf

import java.io.IOException
import java.lang.Byte
import java.lang.Double
import java.lang.Float
import java.lang.Long
import java.lang.Short
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.ReadableByteChannel
import kotlin.ByteArray
import kotlin.Int
import kotlin.Throws
import kotlin.math.min
import kotlin.require

internal class ChannelReader(private val channel: ReadableByteChannel, bufferSize: Int) {
  private val buffer: ByteBuffer
  private var position: Long

  init {
    this.buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.LITTLE_ENDIAN)
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
    ensure(Byte.BYTES)
    position += Byte.BYTES.toLong()
    return buffer.get()
  }

  @Throws(IOException::class)
  fun readShort(): Short {
    ensure(Short.BYTES)
    position += Short.BYTES.toLong()
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
    ensure(Long.BYTES)
    position += Long.BYTES.toLong()
    return buffer.getLong()
  }

  @Throws(IOException::class)
  fun readFloat(): Float {
    return Float.intBitsToFloat(readInt())
  }

  @Throws(IOException::class)
  fun readDouble(): Double {
    return Double.longBitsToDouble(readLong())
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
