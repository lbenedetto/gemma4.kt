package com.llama4j.floattensor

import com.llama4j.gguf.GGMLTensorEntry
import com.llama4j.gguf.GGMLType

object FloatTensorFactory {
  fun create(ggmlType: GGMLType, entry: GGMLTensorEntry): FloatTensor {
    val numElements: Long = FloatTensor.Companion.numberOfElementsLong(*entry.shape)
    return when (ggmlType) {
      GGMLType.Q8_0 -> Q8_0FloatTensor(numElements, entry.memorySegment)
      GGMLType.Q4_0 -> Q4_0FloatTensor(numElements, entry.memorySegment)
      GGMLType.Q4_1 -> Q4_1FloatTensor(numElements, entry.memorySegment)
      GGMLType.Q5_1 -> Q5_1FloatTensor(numElements, entry.memorySegment)
      GGMLType.Q4_K -> Q4_KFloatTensor(numElements, entry.memorySegment)
      GGMLType.Q5_K -> Q5_KFloatTensor(numElements, entry.memorySegment)
      GGMLType.Q6_K -> Q6_KFloatTensor(numElements, entry.memorySegment)
      GGMLType.F32 -> F32FloatTensor(numElements, entry.memorySegment)
      GGMLType.F16 -> F16FloatTensor(numElements, entry.memorySegment)
      GGMLType.BF16 -> BF16FloatTensor(numElements, entry.memorySegment)
      GGMLType.MXFP4 -> MXFP4FloatTensor(numElements, entry.memorySegment)
      else -> throw UnsupportedOperationException("Quantization format " + ggmlType)
    }
  }
}
