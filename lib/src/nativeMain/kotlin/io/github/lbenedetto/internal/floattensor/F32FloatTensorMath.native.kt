package io.github.lbenedetto.internal.floattensor

import ggml.bridge.ggml_bridge_dot_f32_mem
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

@OptIn(ExperimentalForeignApi::class)
actual object F32FloatTensorMath {
  internal actual fun vectorDot(
    thiz: F32FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float {
    val byteOffset = thisOffset.toLong() * Float.SIZE_BYTES
    val quantizedPtr = thiz.memorySegment.rawPointer(byteOffset)
    that.values.usePinned { pinned ->
      return ggml_bridge_dot_f32_mem(
        size,
        quantizedPtr,
        pinned.addressOf(thatOffset)
      )
    }
  }
}
