package io.github.lbenedetto.internal.gguf

import okio.BufferedSource

internal class GgufSource(private val source: BufferedSource) {
  private var _position: Long = 0

  fun position(): Long = _position

  fun readByte(): Byte {
    _position += 1
    return source.readByte()
  }

  fun readShort(): Short {
    _position += 2
    return source.readShortLe()
  }

  fun readInt(): Int {
    _position += 4
    return source.readIntLe()
  }

  fun readLong(): Long {
    _position += 8
    return source.readLongLe()
  }

  fun readFloat(): Float = Float.fromBits(readInt())

  fun readDouble(): Double = Double.fromBits(readLong())

  fun readBytes(length: Int): ByteArray {
    _position += length
    return source.readByteArray(length.toLong())
  }

  fun skipBytes(length: Int) {
    _position += length
    source.skip(length.toLong())
  }
}
