package io.github.lbenedetto.internal.data

expect fun wrapWithFloatBuffer(array: FloatArray): FloatBuffer

expect class FloatBuffer {
  fun remaining(): Int
  fun get(destination: FloatArray): FloatBuffer
  fun get(index: Int) : Float
  fun rewind(): FloatBuffer
}
