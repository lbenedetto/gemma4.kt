package io.github.lbenedetto.internal.util

actual class FloatBuffer(private val array: FloatArray, private var pos: Int = 0) {
  actual fun remaining(): Int = array.size - pos

  actual fun get(destination: FloatArray): FloatBuffer {
    array.copyInto(destination, 0, pos, pos + destination.size)
    pos += destination.size
    return this
  }

  actual fun get(index: Int): Float = array[pos + index]
}

actual fun wrapWithFloatBuffer(array: FloatArray): FloatBuffer = FloatBuffer(array)
