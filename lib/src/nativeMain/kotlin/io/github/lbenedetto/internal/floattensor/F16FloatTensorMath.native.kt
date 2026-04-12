package io.github.lbenedetto.internal.floattensor

import ggml.bridge.ggml_bridge_dot_f16_f32
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

@OptIn(ExperimentalForeignApi::class)
actual object F16FloatTensorMath {
  internal actual fun vectorDot(
    thiz: F16FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float {
    val byteOffset = thisOffset.toLong() * 2 // F16 = 2 bytes per element
    val f16Ptr = thiz.memorySegment.rawPointer(byteOffset)
    that.values.usePinned { pinned ->
      return ggml_bridge_dot_f16_f32(
        size,
        f16Ptr,
        pinned.addressOf(thatOffset)
      )
    }
  }
}
