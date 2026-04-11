package io.github.lbenedetto.internal.floattensor

expect object Q6_KFloatTensorMath {
  internal fun vectorDot(
    thiz: Q6_KFloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
