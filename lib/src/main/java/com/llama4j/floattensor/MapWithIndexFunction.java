package com.llama4j.floattensor;

@FunctionalInterface
public interface MapWithIndexFunction {
  float apply(float value, int index);
}
