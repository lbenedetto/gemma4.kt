package io.github.lbenedetto.internal.floattensor

expect object BF16FloatTensorMath {
  internal fun vectorDot(
    thiz: BF16FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
