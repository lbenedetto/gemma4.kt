package io.github.lbenedetto.internal.floattensor

expect object Q5_KFloatTensorMath {
  internal fun vectorDot(
    thiz: Q5_KFloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
