package io.github.lbenedetto.internal.floattensor

import ggml.bridge.ggml_bridge_dot_q8_0_f32_dequant
import io.github.lbenedetto.internal.floattensor.FloatTensor.Companion.scalarDot
import io.github.lbenedetto.internal.gguf.GGMLType
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import kotlin.math.min

@OptIn(ExperimentalForeignApi::class)
actual object Q8_0FloatTensorMath {
  internal actual fun vectorDot(
    thiz: Q8_0FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float {
    var result = 0f
    var j = 0
    val blockSize = GGMLType.Q8_0.blockSize

    val alignmentBound = min(size, -thisOffset and (blockSize - 1))
    if (alignmentBound > 0) {
      result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound)
      j += alignmentBound
    }

    val fullBlocks = (size - j) / blockSize * blockSize
    if (fullBlocks > 0) {
      val byteOffset = (thisOffset + j).toLong() / blockSize * GGMLType.Q8_0.typeSize
      val quantizedPtr = thiz.memorySegment.rawPointer(byteOffset)
      that.values.usePinned { pinned ->
        result += ggml_bridge_dot_q8_0_f32_dequant(fullBlocks, quantizedPtr, pinned.addressOf(thatOffset + j))
      }
      j += fullBlocks
    }

    if (j < size) {
      result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j)
    }

    return result
  }
}
