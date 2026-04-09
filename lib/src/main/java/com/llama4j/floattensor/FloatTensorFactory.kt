package com.llama4j.floattensor;

import com.llama4j.gguf.GGMLTensorEntry;
import com.llama4j.gguf.GGMLType;

public class FloatTensorFactory {
  public static FloatTensor create(GGMLType ggmlType, GGMLTensorEntry entry) {
    long numElements = FloatTensor.numberOfElementsLong(entry.shape());
    return switch (ggmlType) {
      case Q8_0 -> new Q8_0FloatTensor(numElements, entry.memorySegment());
      case Q4_0 -> new Q4_0FloatTensor(numElements, entry.memorySegment());
      case Q4_1 -> new Q4_1FloatTensor(numElements, entry.memorySegment());
      case Q5_1 -> new Q5_1FloatTensor(numElements, entry.memorySegment());
      case Q4_K -> new Q4_KFloatTensor(numElements, entry.memorySegment());
      case Q5_K -> new Q5_KFloatTensor(numElements, entry.memorySegment());
      case Q6_K -> new Q6_KFloatTensor(numElements, entry.memorySegment());
      case F32 -> new F32FloatTensor(numElements, entry.memorySegment());
      case F16 -> new F16FloatTensor(numElements, entry.memorySegment());
      case BF16 -> new BF16FloatTensor(numElements, entry.memorySegment());
      case MXFP4 -> new MXFP4FloatTensor(numElements, entry.memorySegment());
      default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
    };
  }
}
