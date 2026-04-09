package com.llama4j.floattensor;

import com.llama4j.gguf.GGMLType;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static java.util.Objects.requireNonNull;

final class MXFP4FloatTensor extends FloatTensor {

    private static final int[] MXFP4_VALUES = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    private final long size;
    private final MemorySegment memorySegment;

    MXFP4FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override public long size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.MXFP4; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.QK_MXFP4;
        int inBlockIndex = (int) (index % GGMLType.QK_MXFP4);
        long blockOffset = blockIndex * GGMLType.MXFP4.getTypeSize();

        int e8m0 = Byte.toUnsignedInt(readByte(memorySegment, blockOffset));
        float d = e8m0ToFp32Half(e8m0);

        long qsOffset = blockOffset + Byte.BYTES + (inBlockIndex & 0x0F);
        int packed = Byte.toUnsignedInt(readByte(memorySegment, qsOffset));
        int nibble = inBlockIndex < (GGMLType.QK_MXFP4 / 2) ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);

        return MXFP4_VALUES[nibble] * d;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (that instanceof ArrayFloatTensor aft) {
            if (FloatTensor.USE_VECTOR_API) {
                return vectorDot(this, thisOffset, aft, thatOffset, size);
            }
            return scalarDot(this, thisOffset, aft, thatOffset, size);
        }
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(MXFP4FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert Integer.bitCount(GGMLType.QK_MXFP4) == 1 : "power of 2";
        int j = 0;
        float result = 0f;

        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.QK_MXFP4 - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j = alignmentBound;
        }

        int upperBound = j + (size - j) / GGMLType.QK_MXFP4 * GGMLType.QK_MXFP4;
        for (; j < upperBound; j += GGMLType.QK_MXFP4) {
            assert (thisOffset + j) % GGMLType.QK_MXFP4 == 0;
            long blockOffset = (long) (thisOffset + j) / GGMLType.QK_MXFP4 * GGMLType.MXFP4.getTypeSize();
            float d = e8m0ToFp32Half(Byte.toUnsignedInt(readByte(thiz.memorySegment, blockOffset)));

            ByteVector packed = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Byte.BYTES, ByteOrder.LITTLE_ENDIAN);
            ByteVector lo = packed.and((byte) 0x0F);
            ByteVector hi = packed.lanewise(VectorOperators.LSHR, 4);

            float blockSum = 0f;
            switch (requireNonNull(F_SPECIES).vectorBitSize()) {
                case 512 -> {
                    FloatVector loCoeffs = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, 0));
                    FloatVector hiCoeffs = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, 0));
                    FloatVector xLo = that.getFloatVector(F_SPECIES, thatOffset + j);
                    FloatVector xHi = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2);
                    blockSum += loCoeffs.fma(xLo, hiCoeffs.mul(xHi)).reduceLanes(VectorOperators.ADD);
                }
                case 256 -> {
                    FloatVector lo0 = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, 0));
                    FloatVector lo1 = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, 1));
                    FloatVector hi0 = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, 0));
                    FloatVector hi1 = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, 1));
                    FloatVector x0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    FloatVector x1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    FloatVector x2 = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2);
                    FloatVector x3 = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2 + F_SPECIES.length());
                    blockSum += lo0.fma(x0, lo1.mul(x1)).reduceLanes(VectorOperators.ADD);
                    blockSum += hi0.fma(x2, hi1.mul(x3)).reduceLanes(VectorOperators.ADD);
                }
                case 128 -> {
                    FloatVector sum = FloatVector.zero(F_SPECIES);
                    for (int p = 0; p < 4; p++) {
                        FloatVector loPart = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, p));
                        FloatVector hiPart = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, p));
                        FloatVector xLo = that.getFloatVector(F_SPECIES, thatOffset + j + p * F_SPECIES.length());
                        FloatVector xHi = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2 + p * F_SPECIES.length());
                        sum = loPart.fma(xLo, sum);
                        sum = hiPart.fma(xHi, sum);
                    }
                    blockSum += sum.reduceLanes(VectorOperators.ADD);
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }

            result += blockSum * d;
        }

        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }

    private static FloatVector mxfp4CodesToCoeffs(FloatVector codes) {
        FloatVector zero = FloatVector.zero(F_SPECIES);
        FloatVector eight = FloatVector.broadcast(F_SPECIES, 8f);
        var negMask = codes.compare(VectorOperators.GE, 8f);

        FloatVector t = codes.sub(zero.blend(eight, negMask));
        FloatVector mag = t
                .add(t.sub(4f).lanewise(VectorOperators.MAX, 0f))
                .add(t.sub(6f).lanewise(VectorOperators.MAX, 0f).mul(2f));
        return mag.blend(mag.neg(), negMask);
    }

    private static float scalarDot(MXFP4FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int i = 0; i < size; i++) {
            result += thiz.getFloat(thisOffset + i) * that.values[thatOffset + i];
        }
        return result;
    }

    private static float e8m0ToFp32Half(int x) {
        int bits;
        if (x < 2) {
            bits = 0x00200000 << x;
        } else {
            bits = (x - 1) << 23;
        }
        return Float.intBitsToFloat(bits);
    }
}
