package io.github.lbenedetto.internal.floattensor

expect object MXFP4FloatTensorMath {
  internal fun vectorDot(
    thiz: MXFP4FloatTensor,
    thisOffset: Int,
    that: ArrayFloatTensor,
    thatOffset: Int,
    size: Int
  ): Float
}
