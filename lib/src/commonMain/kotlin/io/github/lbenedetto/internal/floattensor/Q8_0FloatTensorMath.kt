package io.github.lbenedetto.internal.floattensor

expect object Q8_0FloatTensorMath {
  internal fun vectorDot(
    thiz: Q8_0FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
