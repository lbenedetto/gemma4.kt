package io.github.lbenedetto.internal.floattensor

import jdk.incubator.vector.VectorShape

actual fun vectorBitSize(): Int =
  Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize())
