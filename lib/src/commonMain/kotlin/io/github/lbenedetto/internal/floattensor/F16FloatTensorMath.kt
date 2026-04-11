package io.github.lbenedetto.internal.floattensor

expect object F16FloatTensorMath {
  internal fun vectorDot(
    thiz: F16FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
