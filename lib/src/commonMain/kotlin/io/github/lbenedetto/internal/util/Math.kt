package io.github.lbenedetto.internal.util

import kotlin.math.abs

object Math {

  // Copied from java.lang.Math.multiplyExact(long, long)
  fun multiplyExact(x: Long, y: Long): Long {
    val r = x * y
    val ax = abs(x)
    val ay = abs(y)
    if (((ax or ay) ushr 31 != 0L)) {
      // Some bits greater than 2^31 that might cause overflow
      // Check the result using the divide operator
      // and check for the special case of Long.MIN_VALUE * -1
      if (((y != 0L) && (r / y != x)) || (x == Long.MIN_VALUE && y == -1L)) {
        throw ArithmeticException("long overflow")
      }
    }
    return r
  }

  // Copied from java.lang.Math.multiplyExact(int, int)
  fun multiplyExact(x: Int, y: Int): Int {
    val r = x.toLong() * y.toLong()
    if (r.toInt().toLong() != r) {
      throw ArithmeticException("integer overflow")
    }
    return r.toInt()
  }

  fun toIntExact(value: Long): Int {
    if (value.toInt().toLong() != value) {
      throw ArithmeticException("integer overflow")
    }
    return value.toInt()
  }
}
