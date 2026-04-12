package io.github.lbenedetto.internal.floattensor

import ggml.bridge.ggml_bridge_dot_f32
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

@OptIn(ExperimentalForeignApi::class)
actual object ArrayFloatTensorMath {
  internal actual fun vectorDot(
    thiz: ArrayFloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float {
    thiz.values.usePinned { pinnedThis ->
      that.values.usePinned { pinnedThat ->
        return ggml_bridge_dot_f32(
          size,
          pinnedThis.addressOf(thisOffset),
          pinnedThat.addressOf(thatOffset)
        )
      }
    }
  }
}
