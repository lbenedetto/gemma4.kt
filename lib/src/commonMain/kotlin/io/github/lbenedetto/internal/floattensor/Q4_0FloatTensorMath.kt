package io.github.lbenedetto.internal.floattensor

expect object Q4_0FloatTensorMath {
  internal fun vectorDot(
    thiz: Q4_0FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
