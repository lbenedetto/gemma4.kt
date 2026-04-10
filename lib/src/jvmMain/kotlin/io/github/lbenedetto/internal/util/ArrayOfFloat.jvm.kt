package io.github.lbenedetto.internal.util
import java.nio.FloatBuffer as JFloatBuffer

actual fun wrapWithFloatBuffer(array: FloatArray): FloatBuffer = FloatBuffer(JFloatBuffer.wrap(array))

actual class FloatBuffer(
  private val floatBuffer: JFloatBuffer
) {
  actual fun remaining(): Int = floatBuffer.remaining()

  actual fun get(array: FloatArray): FloatBuffer {
    floatBuffer.get(array)
    return this
  }

  actual fun get(index: Int) : Float = floatBuffer.get(index)
}
