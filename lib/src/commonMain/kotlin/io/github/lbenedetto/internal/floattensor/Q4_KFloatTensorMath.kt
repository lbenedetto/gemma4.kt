package io.github.lbenedetto.internal.floattensor

expect object Q4_KFloatTensorMath {
  internal fun vectorDot(
    thiz: Q4_KFloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
