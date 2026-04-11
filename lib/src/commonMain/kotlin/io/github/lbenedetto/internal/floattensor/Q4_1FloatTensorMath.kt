package io.github.lbenedetto.internal.floattensor

expect object Q4_1FloatTensorMath {
  internal fun vectorDot(
    thiz: Q4_1FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
