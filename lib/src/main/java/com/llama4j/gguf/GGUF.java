package com.llama4j.gguf;

import com.llama4j.floattensor.FloatTensor;
import com.llama4j.util.Timer;
import kotlin.Pair;
import org.jspecify.annotations.Nullable;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.util.Objects.requireNonNull;

public final class GGUF {
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32; // must be a power of 2
    private static final int PARSE_BUFFER_SIZE = 1 << 20;
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(2, 3);
    private int tensorCount; // uint64_t
    private int alignment;
    private @Nullable Map<String, Object> metadata; // lateinit
    private @Nullable Map<String, GGUFTensorInfo> tensorInfos; // lateinit
    private long tensorDataOffset;

    public Map<String, Object> getMetadata() {
        return requireNonNull(metadata);
    }

    public long getTensorDataOffset() {
        return tensorDataOffset;
    }

    public Map<String, GGUFTensorInfo> getTensorInfos() {
        return requireNonNull(tensorInfos);
    }

    public static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, long tensorDataOffset, Map<String, GGUFTensorInfo> tensorInfos) throws IOException {
        Arena arena = Arena.global();
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena);
        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensorInfos.size());
        for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUFTensorInfo ti = entry.getValue();
            long numberOfElements = FloatTensor.numberOfElementsLong(ti.dimensions());
            long sizeInBytes = ti.ggmlType().byteSizeFor(numberOfElements);
            MemorySegment memorySegment = tensorData.asSlice(ti.offset(), sizeInBytes);
            tensorEntries.put(ti.name(), new GGMLTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
        return tensorEntries;
    }

    public static GGUF loadModel(Path modelPath) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(modelPath)) {
            return loadModel(fileChannel, modelPath.toString());
        }
    }

    public static GGUF loadModel(FileChannel fileChannel, String modelLabel) throws IOException {
        try (var ignored = Timer.log("Parse " + modelLabel)) {
            fileChannel.position(0L);
            GGUF gguf = new GGUF();
            ChannelReader reader = new ChannelReader(fileChannel, PARSE_BUFFER_SIZE);
            gguf.loadModelImpl(reader);
            return gguf;
        }
    }

    private void loadModelImpl(ChannelReader reader) throws IOException {
        readHeader(reader);
        this.tensorInfos = HashMap.newHashMap(tensorCount);
        for (int i = 0; i < tensorCount; ++i) {
            GGUFTensorInfo ti = readTensorInfo(reader);
            assert !tensorInfos.containsKey(ti.name());
            tensorInfos.put(ti.name(), ti);
        }
        long position = reader.position();
        int padding = (int) ((getAlignment() - (position % getAlignment())) % getAlignment());
        skipBytes(reader, padding);
        this.tensorDataOffset = reader.position();
    }

    private GGMLType readGGMLType(ChannelReader reader) throws IOException {
        int ggmlTypeId = readInt(reader);
        return GGMLType.fromId(ggmlTypeId);
    }

    private GGUFTensorInfo readTensorInfo(ChannelReader reader) throws IOException {
        String name = readString(reader);
        assert name.length() <= 64;
        int n_dimensions = readInt(reader);
        assert n_dimensions <= 4;
        int[] dimensions = new int[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            dimensions[i] = Math.toIntExact(readLong(reader));
        }
        GGMLType ggmlType = readGGMLType(reader);
        long offset = readLong(reader);
        assert offset % getAlignment() == 0;
        return new GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    private String readString(ChannelReader reader) throws IOException {
        int len = Math.toIntExact(readLong(reader));
        return new String(readBytes(reader, len), StandardCharsets.UTF_8);
    }

    private Pair<String, Object> readKeyValuePair(ChannelReader reader) throws IOException {
        String key = readString(reader);
        assert key.length() < (1 << 16);
        assert key.codePoints().allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');
        Object value = readMetadataValue(reader);
        return new Pair<>(key, value);
    }

    private Object readMetadataValue(ChannelReader reader) throws IOException {
        MetadataValueType valueType = readMetadataValueType(reader);
        return readMetadataValueOfType(valueType, reader);
    }

    void readHeader(ChannelReader reader) throws IOException {
        int magic = readInt(reader);
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }
        int version = readInt(reader);
        if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }
        this.tensorCount = Math.toIntExact(readLong(reader));
        // uint64_t
        int metadata_kv_count = Math.toIntExact(readLong(reader));
        this.metadata = HashMap.newHashMap(metadata_kv_count);
        for (int i = 0; i < metadata_kv_count; ++i) {
            Pair<String, Object> keyValue = readKeyValuePair(reader);
            assert !metadata.containsKey(keyValue.getFirst());
            metadata.put(keyValue.getFirst(), keyValue.getSecond());
        }
    }

    private Object readArray(ChannelReader reader) throws IOException {
        MetadataValueType valueType = readMetadataValueType(reader);
        int len = Math.toIntExact(readLong(reader));
        switch (valueType) {
            case UINT8, INT8 -> {
                return readBytes(reader, len);
            }
            case UINT16, INT16 -> {
                short[] shorts = new short[len];
                for (int i = 0; i < len; ++i) {
                    shorts[i] = readShort(reader);
                }
                return shorts;
            }
            case UINT32, INT32 -> {
                int[] ints = new int[len];
                for (int i = 0; i < len; ++i) {
                    ints[i] = readInt(reader);
                }
                return ints;
            }
            case FLOAT32 -> {
                float[] floats = new float[len];
                for (int i = 0; i < len; ++i) {
                    floats[i] = readFloat(reader);
                }
                return floats;
            }
            case BOOL -> {
                boolean[] booleans = new boolean[len];
                for (int i = 0; i < len; ++i) {
                    booleans[i] = readBoolean(reader);
                }
                return booleans;
            }
            case STRING -> {
                String[] strings = new String[len];
                for (int i = 0; i < len; ++i) {
                    strings[i] = readString(reader);
                }
                return strings;
            }
            case ARRAY -> {
                Object[] arrays = new Object[len];
                for (int i = 0; i < len; ++i) {
                    arrays[i] = readArray(reader);
                }
                return arrays;
            }
            default -> throw new UnsupportedOperationException("read array of " + valueType);
        }
    }

    private Object readMetadataValueOfType(MetadataValueType valueType, ChannelReader reader) throws IOException {
        return switch (valueType) {
            case UINT8, INT8 -> readByte(reader);
            case UINT16, INT16 -> readShort(reader);
            case UINT32, INT32 -> readInt(reader);
            case FLOAT32 -> readFloat(reader);
            case UINT64, INT64 -> readLong(reader);
            case FLOAT64 -> readDouble(reader);
            case BOOL -> readBoolean(reader);
            case STRING -> readString(reader);
            case ARRAY -> readArray(reader);
        };
    }

    private MetadataValueType readMetadataValueType(ChannelReader reader) throws IOException {
        int index = readInt(reader);
        return MetadataValueType.fromIndex(index);
    }

    private byte[] readBytes(ChannelReader reader, int length) throws IOException {
        return reader.readBytes(length);
    }

    private void skipBytes(ChannelReader reader, int length) throws IOException {
        reader.skipBytes(length);
    }

    private byte readByte(ChannelReader reader) throws IOException {
        return reader.readByte();
    }

    private boolean readBoolean(ChannelReader reader) throws IOException {
        return readByte(reader) != 0;
    }

    private short readShort(ChannelReader reader) throws IOException {
        return reader.readShort();
    }

    private int readInt(ChannelReader reader) throws IOException {
        return reader.readInt();
    }

    private long readLong(ChannelReader reader) throws IOException {
        return reader.readLong();
    }

    private float readFloat(ChannelReader reader) throws IOException {
        return reader.readFloat();
    }

    private double readDouble(ChannelReader reader) throws IOException {
        return reader.readDouble();
    }

    public int getAlignment() {
        if (alignment != 0) {
            return alignment;
        }
        alignment = (int) requireNonNull(metadata).getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }
}
