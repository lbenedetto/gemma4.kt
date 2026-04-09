package com.llama4j.gguf;

enum MetadataValueType {
  UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64;

  private static final MetadataValueType[] VALUES = values();

  public static MetadataValueType fromIndex(int index) {
    return VALUES[index];
  }
}
