package io.github.lbenedetto.internal.floattensor

import ggml.bridge.ggml_bridge_dot_bf16_f32
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

@OptIn(ExperimentalForeignApi::class)
actual object BF16FloatTensorMath {
  internal actual fun vectorDot(
    thiz: BF16FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float {
    val byteOffset = thisOffset.toLong() * 2 // BF16 = 2 bytes per element
    val bf16Ptr = thiz.memorySegment.rawPointer(byteOffset)
    that.values.usePinned { pinned ->
      return ggml_bridge_dot_bf16_f32(
        size,
        bf16Ptr,
        pinned.addressOf(thatOffset)
      )
    }
  }
}
