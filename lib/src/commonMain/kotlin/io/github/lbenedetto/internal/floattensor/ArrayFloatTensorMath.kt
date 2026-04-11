package io.github.lbenedetto.internal.floattensor

expect object ArrayFloatTensorMath{
  internal fun vectorDot(
    thiz: ArrayFloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
