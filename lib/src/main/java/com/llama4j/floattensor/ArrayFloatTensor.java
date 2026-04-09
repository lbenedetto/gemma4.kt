package com.llama4j.floattensor;

import com.llama4j.gguf.GGMLType;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.nio.FloatBuffer;
import java.util.Arrays;

import static java.util.Objects.requireNonNull;

public final class ArrayFloatTensor extends FloatTensor {

    final long size;
    final float[] values;

    ArrayFloatTensor(float[] values) {
        this.size = values.length;
        this.values = values;
    }

    ArrayFloatTensor(FloatBuffer buf) {
        this.values = new float[buf.remaining()];
        this.size = values.length;
        buf.get(this.values);
        buf.rewind();
    }

    public static FloatTensor allocate(int... dims) {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new ArrayFloatTensor(new float[numberOfElements]);
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public float getFloat(long index) {
        return values[Math.toIntExact(index)];
    }

    @Override
    public void setFloat(int index, float value) {
        values[index] = value;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
        Arrays.fill(values, thisOffset, thisOffset + size, value);
        return this;
    }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromArray(species, values, index);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (that instanceof ArrayFloatTensor aft) {
            if (USE_VECTOR_API) {
                return vectorDot(this, thisOffset, aft, thatOffset, size);
            }
            return FloatTensor.scalarDot(this, thisOffset, aft, thatOffset, size);
        }
        return that.dot(thatOffset, this, thisOffset, size);
    }

    private static float vectorDot(ArrayFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        FloatVector val = FloatVector.zero(requireNonNull(F_SPECIES));
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            var a = FloatVector.fromArray(F_SPECIES, thiz.values, thisOffset + i);
            var b = FloatVector.fromArray(F_SPECIES, that.values, thatOffset + i);
            val = a.fma(b, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        for (int i = upperBound; i < size; i++) {
            result += thiz.values[thisOffset + i] * that.values[thatOffset + i];
        }
        return result;
    }
}
