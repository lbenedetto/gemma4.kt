package io.github.lbenedetto.internal.floattensor

expect object Q5_1FloatTensorMath {
  internal fun vectorDot(
    thiz: Q5_1FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
