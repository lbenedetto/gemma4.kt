package io.github.lbenedetto.internal.floattensor

expect object F32FloatTensorMath {
  internal fun vectorDot(
    thiz: F32FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
