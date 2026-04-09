package com.llama4j;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;

final class Q5_1FloatTensor extends FloatTensor {

    private final long size;
    private final MemorySegment memorySegment;

    Q5_1FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q5_1; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q5_1.getBlockSize();
        int inBlockIndex = (int) (index % GGMLType.Q5_1.getBlockSize());
        long blockOffset = blockIndex * GGMLType.Q5_1.getTypeSize();

        float d = readFloat16(memorySegment, blockOffset);
        float m = readFloat16(memorySegment, blockOffset + Float16.BYTES);
        int qh = readInt32LE(memorySegment, blockOffset + 2L * Float16.BYTES);

        int j;
        int nibble;
        int xh;
        if (inBlockIndex < GGMLType.Q5_1.getBlockSize() / 2) {
            j = inBlockIndex;
            nibble = Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2L * Float16.BYTES + Integer.BYTES + j)) & 0x0F;
            xh = ((qh >> j) << 4) & 0x10;
        } else {
            j = inBlockIndex - GGMLType.Q5_1.getBlockSize() / 2;
            nibble = (Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2L * Float16.BYTES + Integer.BYTES + j)) >>> 4) & 0x0F;
            xh = (qh >> (j + 12)) & 0x10;
        }

        int q = nibble | xh;
        return q * d + m;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (that instanceof ArrayFloatTensor aft) {
            if (FloatTensor.USE_VECTOR_API) {
                return vectorDot(this, thisOffset, aft, thatOffset, size);
            }
            return scalarDot(this, thisOffset, aft, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(Q5_1FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert Integer.bitCount(GGMLType.Q5_1.getBlockSize()) == 1 : "power of 2";
        int j = 0;
        float result = 0f;

        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q5_1.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j = alignmentBound;
        }

        float[] decoded = new float[GGMLType.Q5_1.getBlockSize()];
        int upperBound = j + (size - j) / GGMLType.Q5_1.getBlockSize() * GGMLType.Q5_1.getBlockSize();
        int vecUpper = F_SPECIES.loopBound(GGMLType.Q5_1.getBlockSize());
        for (; j < upperBound; j += GGMLType.Q5_1.getBlockSize()) {
            assert (thisOffset + j) % GGMLType.Q5_1.getBlockSize() == 0;
            long blockOffset = (long) (thisOffset + j) / GGMLType.Q5_1.getBlockSize() * GGMLType.Q5_1.getTypeSize();
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float m = readFloat16(thiz.memorySegment, blockOffset + Float16.BYTES);
            int qh = readInt32LE(thiz.memorySegment, blockOffset + 2L * Float16.BYTES);
            long qsBase = blockOffset + 2L * Float16.BYTES + Integer.BYTES;

            for (int p = 0; p < GGMLType.Q5_1.getBlockSize() / 2; p++) {
                int packed = Byte.toUnsignedInt(readByte(thiz.memorySegment, qsBase + p));
                int x0 = (packed & 0x0F) | ((((qh >> p) << 4) & 0x10));
                int x1 = ((packed >>> 4) & 0x0F) | ((qh >> (p + 12)) & 0x10);
                decoded[p] = x0 * d + m;
                decoded[p + GGMLType.Q5_1.getBlockSize() / 2] = x1 * d + m;
            }

            FloatVector acc = FloatVector.zero(F_SPECIES);
            for (int i = 0; i < vecUpper; i += F_SPECIES.length()) {
                FloatVector w = FloatVector.fromArray(F_SPECIES, decoded, i);
                FloatVector x = that.getFloatVector(F_SPECIES, thatOffset + j + i);
                acc = w.fma(x, acc);
            }
            result += acc.reduceLanes(VectorOperators.ADD);

            for (int i = vecUpper; i < GGMLType.Q5_1.getBlockSize(); i++) {
                result += decoded[i] * that.values[thatOffset + j + i];
            }
        }

        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }

    private static float scalarDot(Q5_1FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int i = 0; i < size; i++) {
            result += thiz.getFloat(thisOffset + i) * that.values[thatOffset + i];
        }
        return result;
    }

    private static int readInt32LE(MemorySegment memorySegment, long offset) {
        int b0 = Byte.toUnsignedInt(readByte(memorySegment, offset));
        int b1 = Byte.toUnsignedInt(readByte(memorySegment, offset + 1));
        int b2 = Byte.toUnsignedInt(readByte(memorySegment, offset + 2));
        int b3 = Byte.toUnsignedInt(readByte(memorySegment, offset + 3));
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}
