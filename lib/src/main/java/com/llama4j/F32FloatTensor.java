package com.llama4j;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

final class F32FloatTensor extends FloatTensor {

    private final long size;
    private final MemorySegment memorySegment;

    F32FloatTensor(long numElements, MemorySegment memorySegment) {
        this.size = numElements;
        this.memorySegment = memorySegment;
    }

    @Override public long size() { return size; }

    @Override
    public float getFloat(long index) {
        return memorySegment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, (long) index * Float.BYTES);
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("read-only");
    }

    @Override public GGMLType type() { return GGMLType.F32; }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromMemorySegment(species, memorySegment, (long) index * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (that instanceof ArrayFloatTensor aft && USE_VECTOR_API) {
            return vectorDot(this, thisOffset, aft, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(F32FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            var a = FloatVector.fromMemorySegment(F_SPECIES, thiz.memorySegment, (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            var b = FloatVector.fromArray(F_SPECIES, that.values, thatOffset + i);
            val = a.fma(b, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        for (int i = upperBound; i < size; i++) {
            result += thiz.getFloat(thisOffset + i) * that.values[thatOffset + i];
        }
        return result;
    }
}
