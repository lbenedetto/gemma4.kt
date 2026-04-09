package com.llama4j.floattensor;

import com.llama4j.gguf.GGMLType;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static java.util.Objects.requireNonNull;

final class BF16FloatTensor extends FloatTensor {

    final long size;
    final MemorySegment memorySegment;

    public BF16FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override public long size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.BF16; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        short bits = readShort(memorySegment, (long) index * 2);
        return Float.intBitsToFloat(bits << 16);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(BF16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert requireNonNull(S_SPECIES_HALF).length() == requireNonNull(F_SPECIES).length();
        FloatVector val = FloatVector.zero(requireNonNull(F_SPECIES));
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bfloat16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment, (thisOffset + i) * 2L, ByteOrder.LITTLE_ENDIAN);
            FloatVector thizVector = bfloat16
                    .castShape(requireNonNull(I_SPECIES), 0)
                    .lanewise(VectorOperators.LSHL, 16)
                    .reinterpretAsFloats();
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }
        return result;
    }
}
