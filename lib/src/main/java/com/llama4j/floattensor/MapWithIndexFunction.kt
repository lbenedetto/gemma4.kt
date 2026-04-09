package com.llama4j.floattensor

fun interface MapWithIndexFunction {
  fun apply(value: Float, index: Int): Float
}
