package io.github.lbenedetto.internal.floattensor

val VECTOR_BIT_SIZE: Int = vectorBitSize()
val USE_VECTOR_API: Boolean = VECTOR_BIT_SIZE != 0

expect fun vectorBitSize(): Int
