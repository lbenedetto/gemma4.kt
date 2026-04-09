package com.llama4j.floattensor;

import com.llama4j.gguf.GGMLType;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static java.util.Objects.requireNonNull;

final class Q4_KFloatTensor extends FloatTensor {

    static final int BLOCK_SIZE = GGMLType.QK_K;
    static final int TYPE_SIZE = GGMLType.Q4_K.getTypeSize();

    final long size;
    final MemorySegment memorySegment;

    public Q4_KFloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override public long size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q4_K; }

    // Decode scale or min for sub-block j (0..7) from the 12-byte scales array
    static int getScaleMinK4(int j, MemorySegment mem, long scalesOffset, boolean isMin) {
        if (j < 4) {
            int idx = isMin ? j + 4 : j;
            return Byte.toUnsignedInt(readByte(mem, scalesOffset + idx)) & 63;
        } else {
            int lowIdx = j + 4;
            int highIdx = isMin ? j : j - 4;
            int low = isMin
                    ? (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) >> 4)
                    : (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) & 0xF);
            int high = (Byte.toUnsignedInt(readByte(mem, scalesOffset + highIdx)) >> 6) & 0x3;
            return low | (high << 4);
        }
    }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        float d = readFloat16(memorySegment, blockOffset);
        float dmin = readFloat16(memorySegment, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qsOffset = blockOffset + 16; // 4 + 12

        // Each group of 64 values uses 2 sub-blocks: low nibble (32) + high nibble (32)
        int group = withinBlock / 64;   // 0..3
        int inGroup = withinBlock % 64;
        int subBlock;
        int nibbleIndex;
        boolean isHigh;
        if (inGroup < 32) {
            subBlock = group * 2;
            nibbleIndex = inGroup;
            isHigh = false;
        } else {
            subBlock = group * 2 + 1;
            nibbleIndex = inGroup - 32;
            isHigh = true;
        }

        int sc = getScaleMinK4(subBlock, memorySegment, scalesOffset, false);
        int m = getScaleMinK4(subBlock, memorySegment, scalesOffset, true);

        byte qsByte = readByte(memorySegment, qsOffset + group * 32 + nibbleIndex);
        int quant = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);

        return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q4_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Handle unaligned head
        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector val = FloatVector.zero(requireNonNull(F_SPECIES));
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float dmin = readFloat16(thiz.memorySegment, blockOffset + 2);
            long scalesOff = blockOffset + 4;
            long qsOff = blockOffset + 16;

            // 4 groups of 64 values each (2 sub-blocks per group: low nibble + high nibble)
            for (int g = 0; g < 4; g++) {
                float d1 = d * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, false);
                float negM1 = -(dmin * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, true));
                float d2 = d * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, false);
                float negM2 = -(dmin * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, true));

                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, negM1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, negM2);

                int loBase = thatOffset + j + g * 64;
                int hiBase = thatOffset + j + g * 64 + 32;

                // Process 32 bytes of qs in 2 chunks of 16 bytes
                for (int c = 0; c < 2; c++) {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qsOff + (long) g * 32 + c * 16, ByteOrder.LITTLE_ENDIAN);
                    var loBytes = wBytes.and((byte) 0xF);
                    var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);

                    int loIdx = loBase + c * 16;
                    int hiIdx = hiBase + c * 16;

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQ = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQ.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx), val);
                            var hiQ = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 = hiQ.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx), val2);
                        }
                        case 256 -> {
                            var loQ0 = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQ1 = loBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQ0.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx), val);
                            val2 = loQ1.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx + F_SPECIES.length()), val2);
                            var hiQ0 = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQ1 = hiBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = hiQ0.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx), val);
                            val2 = hiQ1.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx + F_SPECIES.length()), val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                var loQ = loBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = loQ.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx + p * F_SPECIES.length()), val);
                                var hiQ = hiBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = hiQ.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx + p * F_SPECIES.length()), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        result += val.add(val2).reduceLanes(VectorOperators.ADD);

        // Handle tail
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}
