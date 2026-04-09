package com.llama4j.sampler;

import com.llama4j.floattensor.FloatTensor;

@FunctionalInterface
public interface Sampler {
    int sampleToken(FloatTensor logits);

    Sampler ARGMAX = FloatTensor::argmax;
}
