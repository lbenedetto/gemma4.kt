package com.llama4j.floattensor;

import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;
import org.jspecify.annotations.Nullable;

class VectorSpeciesConfig {
    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    public final VectorSpecies<Float> FLOAT = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
    public final VectorSpecies<Integer> INT = FLOAT.withLanes(int.class);
    public final VectorSpecies<Short> SHORT_HALF = VectorShape.forBitSize(FLOAT.vectorBitSize() / 2).withLanes(short.class);

    private VectorSpeciesConfig() {
    }

    public static @Nullable VectorSpeciesConfig create() {
        if (!USE_VECTOR_API) return null;
        var config = new VectorSpeciesConfig();
        assert config.FLOAT.length() == config.SHORT_HALF.length();
        return config;
    }

}
