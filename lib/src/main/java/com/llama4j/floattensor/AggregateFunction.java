package com.llama4j.floattensor;

@FunctionalInterface
public interface AggregateFunction {
  float apply(float acc, float value);
}
