package com.llama4j.gguf;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.ReadableByteChannel;

final class ChannelReader {
  private final ReadableByteChannel channel;
  private final ByteBuffer buffer;
  private long position;

  ChannelReader(ReadableByteChannel channel, int bufferSize) {
    this.channel = channel;
    this.buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.LITTLE_ENDIAN);
    this.buffer.limit(0);
    this.position = 0L;
  }

  long position() {
    return position;
  }

  private void ensure(int required) throws IOException {
    if (required > buffer.capacity()) {
      throw new IllegalArgumentException("Requested read " + required + " exceeds buffer capacity " + buffer.capacity());
    }
    if (buffer.remaining() >= required) {
      return;
    }
    buffer.compact();
    while (buffer.position() < required) {
      int read = channel.read(buffer);
      if (read < 0) {
        throw new IOException("Unexpected EOF while reading GGUF metadata");
      }
    }
    buffer.flip();
  }

  byte readByte() throws IOException {
    ensure(Byte.BYTES);
    position += Byte.BYTES;
    return buffer.get();
  }

  short readShort() throws IOException {
    ensure(Short.BYTES);
    position += Short.BYTES;
    return buffer.getShort();
  }

  int readInt() throws IOException {
    ensure(Integer.BYTES);
    position += Integer.BYTES;
    return buffer.getInt();
  }

  long readLong() throws IOException {
    ensure(Long.BYTES);
    position += Long.BYTES;
    return buffer.getLong();
  }

  float readFloat() throws IOException {
    return Float.intBitsToFloat(readInt());
  }

  double readDouble() throws IOException {
    return Double.longBitsToDouble(readLong());
  }

  byte[] readBytes(int length) throws IOException {
    byte[] bytes = new byte[length];
    int copied = 0;
    while (copied < length) {
      if (!buffer.hasRemaining()) {
        ensure(1);
      }
      int chunk = Math.min(length - copied, buffer.remaining());
      buffer.get(bytes, copied, chunk);
      copied += chunk;
      position += chunk;
    }
    return bytes;
  }

  void skipBytes(int length) throws IOException {
    int remaining = length;
    while (remaining > 0) {
      if (!buffer.hasRemaining()) {
        ensure(1);
      }
      int chunk = Math.min(remaining, buffer.remaining());
      buffer.position(buffer.position() + chunk);
      remaining -= chunk;
      position += chunk;
    }
  }
}
