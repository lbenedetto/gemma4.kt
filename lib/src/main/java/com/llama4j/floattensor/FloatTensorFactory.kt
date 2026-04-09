package com.llama4j.floattensor

object FloatTensorFactory {
  fun create(ggmlType: com.llama4j.gguf.GGMLType, entry: com.llama4j.gguf.GGMLTensorEntry): FloatTensor {
    val numElements: Long = FloatTensor.numberOfElementsLong(*entry.shape)
    return when (ggmlType) {
      _root_ide_package_.com.llama4j.gguf.GGMLType.Q8_0 -> Q8_0FloatTensor(numElements, entry.memorySegment)
      _root_ide_package_.com.llama4j.gguf.GGMLType.Q4_0 -> Q4_0FloatTensor(numElements, entry.memorySegment)
      _root_ide_package_.com.llama4j.gguf.GGMLType.Q4_1 -> Q4_1FloatTensor(numElements, entry.memorySegment)
      _root_ide_package_.com.llama4j.gguf.GGMLType.Q5_1 -> Q5_1FloatTensor(numElements, entry.memorySegment)
      _root_ide_package_.com.llama4j.gguf.GGMLType.Q4_K -> Q4_KFloatTensor(numElements, entry.memorySegment)
      _root_ide_package_.com.llama4j.gguf.GGMLType.Q5_K -> Q5_KFloatTensor(numElements, entry.memorySegment)
      _root_ide_package_.com.llama4j.gguf.GGMLType.Q6_K -> Q6_KFloatTensor(numElements, entry.memorySegment)
      _root_ide_package_.com.llama4j.gguf.GGMLType.F32 -> F32FloatTensor(numElements, entry.memorySegment)
      _root_ide_package_.com.llama4j.gguf.GGMLType.F16 -> F16FloatTensor(numElements, entry.memorySegment)
      _root_ide_package_.com.llama4j.gguf.GGMLType.BF16 -> BF16FloatTensor(numElements, entry.memorySegment)
      _root_ide_package_.com.llama4j.gguf.GGMLType.MXFP4 -> MXFP4FloatTensor(numElements, entry.memorySegment)
      else -> throw UnsupportedOperationException("Quantization format $ggmlType")
    }
  }
}
