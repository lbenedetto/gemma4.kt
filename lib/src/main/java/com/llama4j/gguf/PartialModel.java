package com.llama4j.gguf;

import com.llama4j.model.Llama;
import com.llama4j.util.Pair;

import java.util.Map;

record PartialModel(
    String modelFileName,
    Llama model,
    long tensorDataOffset,
    Map<String, GGUFTensorInfo> tensorInfos,
    Pair<float[], float[]> ropeFreqsSWA,
    Pair<float[], float[]> ropeFreqsFull
) {
}
