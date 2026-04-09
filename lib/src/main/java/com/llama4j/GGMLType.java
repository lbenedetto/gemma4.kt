package com.llama4j;

enum GGMLType {
    F32(Float.BYTES),          // 0
    F16(Float16.BYTES),        // 1
    Q4_0(Float16.BYTES + 16 * Byte.BYTES, 32),  // 2
    Q4_1(2 * Float16.BYTES + 16 * Byte.BYTES, 32), // 3
    UNSUPPORTED_Q4_2(Integer.MAX_VALUE), // 4 - removed
    UNSUPPORTED_Q4_3(Integer.MAX_VALUE), // 5 - removed
    Q5_0(Integer.MAX_VALUE),   // 6
    Q5_1(2 * Float16.BYTES + Integer.BYTES + 16 * Byte.BYTES, 32),   // 7
    Q8_0(Float16.BYTES + 32 * Byte.BYTES, 32),  // 8
    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32), // 9
    Q2_K(Integer.MAX_VALUE),   // 10
    Q3_K(Integer.MAX_VALUE),   // 11
    Q4_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K), // 12
    Q5_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2, GGMLType.QK_K), // 13
    Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + Float16.BYTES, GGMLType.QK_K), // 14
    Q8_K(Integer.MAX_VALUE),   // 15
    IQ2_XXS(Integer.MAX_VALUE), // 16
    IQ2_XS(Integer.MAX_VALUE),  // 17
    IQ3_XXS(Integer.MAX_VALUE), // 18
    IQ1_S(Integer.MAX_VALUE),   // 19
    IQ4_NL(Integer.MAX_VALUE),  // 20
    IQ3_S(Integer.MAX_VALUE),   // 21
    IQ2_S(Integer.MAX_VALUE),   // 22
    IQ4_XS(Integer.MAX_VALUE),  // 23
    I8(Byte.BYTES),             // 24
    I16(Short.BYTES),           // 25
    I32(Integer.BYTES),         // 26
    I64(Long.BYTES),            // 27
    F64(Double.BYTES),          // 28
    IQ1_M(Integer.MAX_VALUE),   // 29
    BF16(Float16.BYTES),        // 30
    UNSUPPORTED_Q4_0_4_4(Integer.MAX_VALUE), // 31 - removed from gguf files
    UNSUPPORTED_Q4_0_4_8(Integer.MAX_VALUE), // 32
    UNSUPPORTED_Q4_0_8_8(Integer.MAX_VALUE), // 33
    TQ1_0(Integer.MAX_VALUE),   // 34
    TQ2_0(Integer.MAX_VALUE),   // 35
    UNSUPPORTED_IQ4_NL_4_4(Integer.MAX_VALUE), // 36
    UNSUPPORTED_IQ4_NL_4_8(Integer.MAX_VALUE), // 37
    UNSUPPORTED_IQ4_NL_8_8(Integer.MAX_VALUE), // 38
    MXFP4(Byte.BYTES + GGMLType.QK_MXFP4 / 2, GGMLType.QK_MXFP4), // 39
    NVFP4(Integer.MAX_VALUE);   // 40

    private static final GGMLType[] VALUES = values();

    private final int typeSize;

    private final int blockSize;

    public int getTypeSize() {
        return typeSize;
    }

    public int getBlockSize() {
        return blockSize;
    }

    public static GGMLType fromId(int id) {
        if (0 <= id && id < VALUES.length) {
            return VALUES[id];
        }
        throw new UnsupportedOperationException("Unsupported GGML tensor type id: " + id);
    }

    GGMLType(int typeSize) {
        this(typeSize, 1);
    }

    public long byteSizeFor(long numberOfElements) {
        long t = numberOfElements * (long) getTypeSize();
        assert t % getBlockSize() == 0;
        return t / getBlockSize();
    }

    public static final int QK_K = 256;
    public static final int QK_MXFP4 = 32;

    GGMLType(int typeSize, int blockSize) {
        assert blockSize > 0;
        assert typeSize > 0;
        assert isPowerOf2(blockSize);
        this.typeSize = typeSize;
        this.blockSize = blockSize;
    }

    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}
