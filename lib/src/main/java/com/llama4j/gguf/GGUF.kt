package com.llama4j.gguf

import com.llama4j.floattensor.FloatTensor
import java.io.IOException
import java.lang.foreign.Arena
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import java.util.*
import java.util.function.IntPredicate

class GGUF {
  private var tensorCount = 0 // uint64_t
  var alignment: Int = 0
    get() {
      if (field != 0) {
        return field
      }
      field = Objects.requireNonNull<MutableMap<String, Any>?>(metadata)
        .getOrDefault("general.alignment", DEFAULT_ALIGNMENT) as Int
      assert(Integer.bitCount(field) == 1) { "alignment must be a power of two" }
      return field
    }
    private set
  private var metadata: MutableMap<String, Any>? = null // lateinit
  private var tensorInfos: MutableMap<String, GGUFTensorInfo>? = null // lateinit
  var tensorDataOffset: Long = 0
    private set

  fun getMetadata(): MutableMap<String, Any> {
    return Objects.requireNonNull<MutableMap<String, Any>?>(metadata)
  }

  fun getTensorInfos(): MutableMap<String, GGUFTensorInfo> {
    return Objects.requireNonNull<MutableMap<String, GGUFTensorInfo>?>(tensorInfos)
  }

  @Throws(IOException::class)
  private fun loadModelImpl(reader: ChannelReader) {
    readHeader(reader)
    this.tensorInfos = HashMap.newHashMap<String, GGUFTensorInfo>(tensorCount)
    for (i in 0..<tensorCount) {
      val ti = readTensorInfo(reader)
      assert(!tensorInfos!!.containsKey(ti.name))
      tensorInfos!!.put(ti.name, ti)
    }
    val position = reader.position()
    val padding = ((this.alignment - (position % this.alignment)) % this.alignment).toInt()
    skipBytes(reader, padding)
    this.tensorDataOffset = reader.position()
  }

  @Throws(IOException::class)
  private fun readGGMLType(reader: ChannelReader): GGMLType {
    val ggmlTypeId = readInt(reader)
    return GGMLType.Companion.fromId(ggmlTypeId)
  }

  @Throws(IOException::class)
  private fun readTensorInfo(reader: ChannelReader): GGUFTensorInfo {
    val name = readString(reader)
    assert(name.length <= 64)
    val n_dimensions = readInt(reader)
    assert(n_dimensions <= 4)
    val dimensions = IntArray(n_dimensions)
    for (i in 0..<n_dimensions) {
      dimensions[i] = Math.toIntExact(readLong(reader))
    }
    val ggmlType = readGGMLType(reader)
    val offset = readLong(reader)
    assert(offset % this.alignment == 0L)
    return GGUFTensorInfo(name, dimensions, ggmlType, offset)
  }

  @Throws(IOException::class)
  private fun readString(reader: ChannelReader): String {
    val len = Math.toIntExact(readLong(reader))
    return String(readBytes(reader, len), StandardCharsets.UTF_8)
  }

  @Throws(IOException::class)
  private fun readKeyValuePair(reader: ChannelReader): Pair<String, Any> {
    val key = readString(reader)
    assert(key.length < (1 shl 16))
    assert(
      key.codePoints()
        .allMatch(IntPredicate { cp: Int -> ('a'.code <= cp && cp <= 'z'.code) || ('0'.code <= cp && cp <= '9'.code) || cp == '_'.code || cp == '.'.code })
    )
    val value = readMetadataValue(reader)
    return Pair<String, Any>(key, value)
  }

  @Throws(IOException::class)
  private fun readMetadataValue(reader: ChannelReader): Any {
    val valueType = readMetadataValueType(reader)
    return readMetadataValueOfType(valueType, reader)
  }

  @Throws(IOException::class)
  fun readHeader(reader: ChannelReader) {
    val magic = readInt(reader)
    require(magic == GGUF_MAGIC) { "unsupported header.magic " + magic }
    val version = readInt(reader)
    require(SUPPORTED_GGUF_VERSIONS.contains(version)) { "unsupported header.version " + version }
    this.tensorCount = Math.toIntExact(readLong(reader))
    // uint64_t
    val metadata_kv_count = Math.toIntExact(readLong(reader))
    this.metadata = HashMap.newHashMap<String, Any>(metadata_kv_count)
    for (i in 0..<metadata_kv_count) {
      val keyValue = readKeyValuePair(reader)
      assert(!metadata!!.containsKey(keyValue.first))
      metadata!!.put(keyValue.first, keyValue.second)
    }
  }

  @Throws(IOException::class)
  private fun readArray(reader: ChannelReader): Any {
    val valueType = readMetadataValueType(reader)
    val len = Math.toIntExact(readLong(reader))
    when (valueType) {
      MetadataValueType.UINT8, MetadataValueType.INT8 -> {
        return readBytes(reader, len)
      }

      MetadataValueType.UINT16, MetadataValueType.INT16 -> {
        val shorts = ShortArray(len)
        for (i in 0..<len) {
          shorts[i] = readShort(reader)
        }
        return shorts
      }

      MetadataValueType.UINT32, MetadataValueType.INT32 -> {
        val ints = IntArray(len)
        for (i in 0..<len) {
          ints[i] = readInt(reader)
        }
        return ints
      }

      MetadataValueType.FLOAT32 -> {
        val floats = FloatArray(len)
        for (i in 0..<len) {
          floats[i] = readFloat(reader)
        }
        return floats
      }

      MetadataValueType.BOOL -> {
        val booleans = BooleanArray(len)
        for (i in 0..<len) {
          booleans[i] = readBoolean(reader)
        }
        return booleans
      }

      MetadataValueType.STRING -> {
        val strings: Array<String> = arrayOfNulls<String>(len)
        for (i in 0..<len) {
          strings[i] = readString(reader)
        }
        return strings
      }

      MetadataValueType.ARRAY -> {
        val arrays: Array<Any> = arrayOfNulls<Any>(len)
        for (i in 0..<len) {
          arrays[i] = readArray(reader)
        }
        return arrays
      }

      else -> throw UnsupportedOperationException("read array of " + valueType)
    }
  }

  @Throws(IOException::class)
  private fun readMetadataValueOfType(valueType: MetadataValueType, reader: ChannelReader): Any {
    return when (valueType) {
      MetadataValueType.UINT8, MetadataValueType.INT8 -> readByte(reader)
      MetadataValueType.UINT16, MetadataValueType.INT16 -> readShort(reader)
      MetadataValueType.UINT32, MetadataValueType.INT32 -> readInt(reader)
      MetadataValueType.FLOAT32 -> readFloat(reader)
      MetadataValueType.UINT64, MetadataValueType.INT64 -> readLong(reader)
      MetadataValueType.FLOAT64 -> readDouble(reader)
      MetadataValueType.BOOL -> readBoolean(reader)
      MetadataValueType.STRING -> readString(reader)
      MetadataValueType.ARRAY -> readArray(reader)
    }
  }

  @Throws(IOException::class)
  private fun readMetadataValueType(reader: ChannelReader): MetadataValueType {
    val index = readInt(reader)
    return MetadataValueType.Companion.fromIndex(index)
  }

  @Throws(IOException::class)
  private fun readBytes(reader: ChannelReader, length: Int): ByteArray {
    return reader.readBytes(length)
  }

  @Throws(IOException::class)
  private fun skipBytes(reader: ChannelReader, length: Int) {
    reader.skipBytes(length)
  }

  @Throws(IOException::class)
  private fun readByte(reader: ChannelReader): Byte {
    return reader.readByte()
  }

  @Throws(IOException::class)
  private fun readBoolean(reader: ChannelReader): Boolean {
    return readByte(reader).toInt() != 0
  }

  @Throws(IOException::class)
  private fun readShort(reader: ChannelReader): Short {
    return reader.readShort()
  }

  @Throws(IOException::class)
  private fun readInt(reader: ChannelReader): Int {
    return reader.readInt()
  }

  @Throws(IOException::class)
  private fun readLong(reader: ChannelReader): Long {
    return reader.readLong()
  }

  @Throws(IOException::class)
  private fun readFloat(reader: ChannelReader): Float {
    return reader.readFloat()
  }

  @Throws(IOException::class)
  private fun readDouble(reader: ChannelReader): Double {
    return reader.readDouble()
  }

  companion object {
    private const val GGUF_MAGIC = 0x46554747
    private const val DEFAULT_ALIGNMENT = 32 // must be a power of 2
    private val PARSE_BUFFER_SIZE = 1 shl 20
    private val SUPPORTED_GGUF_VERSIONS = mutableListOf<Int>(2, 3)

    @Throws(IOException::class)
    fun loadTensors(
      fileChannel: FileChannel,
      tensorDataOffset: Long,
      tensorInfos: MutableMap<String, GGUFTensorInfo>
    ): MutableMap<String, GGMLTensorEntry> {
      val arena = Arena.global()
      val tensorData =
        fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena)
      val tensorEntries: MutableMap<String, GGMLTensorEntry> =
        HashMap.newHashMap<String, GGMLTensorEntry>(tensorInfos.size)
      for (entry in tensorInfos.entries) {
        val ti = entry.value
        val numberOfElements: Long = FloatTensor.Companion.numberOfElementsLong(*ti.dimensions)
        val sizeInBytes = ti.ggmlType.byteSizeFor(numberOfElements)
        val memorySegment = tensorData.asSlice(ti.offset, sizeInBytes)
        tensorEntries.put(ti.name, GGMLTensorEntry(tensorData, ti.name, ti.ggmlType, ti.dimensions, memorySegment))
      }
      return tensorEntries
    }

    @Throws(IOException::class)
    fun loadModel(modelPath: Path): GGUF {
      FileChannel.open(modelPath).use { fileChannel ->
        return loadModel(fileChannel, modelPath.toString())
      }
    }

    @Throws(IOException::class)
    fun loadModel(fileChannel: FileChannel, modelLabel: String): GGUF {
      log("Parse " + modelLabel).use { ignored ->
        fileChannel.position(0L)
        val gguf = GGUF()
        val reader = ChannelReader(fileChannel, PARSE_BUFFER_SIZE)
        gguf.loadModelImpl(reader)
        return gguf
      }
    }
  }
}
