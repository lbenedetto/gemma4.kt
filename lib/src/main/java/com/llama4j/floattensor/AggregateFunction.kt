package com.llama4j.floattensor

fun interface AggregateFunction {
  fun apply(acc: Float, value: Float): Float
}
