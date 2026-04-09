package com.llama4j.gguf;

public record GGUFTensorInfo(
    String name,
    int[] dimensions,
    GGMLType ggmlType,
    long offset
) {
}
