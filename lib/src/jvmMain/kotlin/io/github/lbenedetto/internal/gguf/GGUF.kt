package io.github.lbenedetto.internal.gguf

import io.github.lbenedetto.internal.floattensor.FloatTensor
import io.github.lbenedetto.internal.util.Math
import io.github.lbenedetto.internal.util.Timer
import io.github.lbenedetto.internal.util.assert
import io.github.lbenedetto.internal.util.toCodePoints
import okio.internal.commonToUtf8String
import java.io.IOException
import java.lang.foreign.Arena
import java.nio.channels.FileChannel
import java.nio.file.Path

internal class GGUF private constructor(reader: GgufSource) {
  private var tensorCount = 0 // uint64_t
  var alignment: Int = 0
    get() {
      if (field != 0) {
        return field
      }
      field = metadata
        .getOrElse("general.alignment") { DEFAULT_ALIGNMENT } as Int
      assert(field.countOneBits() == 1) { "alignment must be a power of two" }
      return field
    }
    private set
  lateinit var metadata: MutableMap<String, Any> // Is definitely initialized in init -> readHeader
  var tensorInfos: MutableMap<String, GGUFTensorInfo>
  var tensorDataOffset: Long = 0
    private set

  init {
    readHeader(reader)
    this.tensorInfos = HashMap(tensorCount)
    repeat((0..<tensorCount).count()) {
      val ti = readTensorInfo(reader)
      assert(!tensorInfos.containsKey(ti.name))
      tensorInfos[ti.name] = ti
    }
    val position = reader.position()
    val padding = ((this.alignment - (position % this.alignment)) % this.alignment).toInt()
    skipBytes(reader, padding)
    this.tensorDataOffset = reader.position()
  }

  private fun readGGMLType(reader: GgufSource): GGMLType {
    val ggmlTypeId = readInt(reader)
    return GGMLType.fromId(ggmlTypeId)
  }

  private fun readTensorInfo(reader: GgufSource): GGUFTensorInfo {
    val name = readString(reader)
    assert(name.length <= 64)
    val nDimensions = readInt(reader)
    assert(nDimensions <= 4)
    val dimensions = IntArray(nDimensions)
    for (i in 0..<nDimensions) {
      dimensions[i] = Math.toIntExact(readLong(reader))
    }
    val ggmlType = readGGMLType(reader)
    val offset = readLong(reader)
    assert(offset % this.alignment == 0L)
    return GGUFTensorInfo(name, dimensions, ggmlType, offset)
  }

  private fun readString(reader: GgufSource): String {
    val len = Math.toIntExact(readLong(reader))
    return readBytes(reader, len).commonToUtf8String()
  }

  private fun readKeyValuePair(reader: GgufSource): Pair<String, Any> {
    val key = readString(reader)
    assert(key.length < (1 shl 16))
    assert(
      key.toCodePoints()
        .all { cp: Int -> ('a'.code <= cp && cp <= 'z'.code) || ('0'.code <= cp && cp <= '9'.code) || cp == '_'.code || cp == '.'.code }
    )
    val value = readMetadataValue(reader)
    return Pair(key, value)
  }

  private fun readMetadataValue(reader: GgufSource): Any {
    val valueType = readMetadataValueType(reader)
    return readMetadataValueOfType(valueType, reader)
  }

  private fun readHeader(reader: GgufSource) {
    val magic = readInt(reader)
    require(magic == GGUF_MAGIC) { "unsupported header.magic $magic" }
    val version = readInt(reader)
    require(SUPPORTED_GGUF_VERSIONS.contains(version)) { "unsupported header.version $version" }
    this.tensorCount = Math.toIntExact(readLong(reader))
    // uint64_t
    val metadataKvCount = Math.toIntExact(readLong(reader))
    this.metadata = HashMap(metadataKvCount)
    repeat((0..<metadataKvCount).count()) {
      val keyValue = readKeyValuePair(reader)
      assert(!metadata.containsKey(keyValue.first))
      metadata[keyValue.first] = keyValue.second
    }
  }

  private fun readArray(reader: GgufSource): Any {
    val valueType = readMetadataValueType(reader)
    val len = Math.toIntExact(readLong(reader))
    return when (valueType) {
      MetadataValueType.UINT8, MetadataValueType.INT8 -> readBytes(reader, len)
      MetadataValueType.UINT16, MetadataValueType.INT16 -> ShortArray(len) { readShort(reader) }
      MetadataValueType.UINT32, MetadataValueType.INT32 -> IntArray(len) { readInt(reader) }
      MetadataValueType.FLOAT32 -> FloatArray(len) { readFloat(reader) }
      MetadataValueType.BOOL -> BooleanArray(len) { readBoolean(reader) }
      MetadataValueType.STRING -> Array(len) { readString(reader) }
      MetadataValueType.ARRAY -> Array(len) { readArray(reader) }
      else -> throw UnsupportedOperationException("read array of $valueType")
    }
  }

  private fun readMetadataValueOfType(valueType: MetadataValueType, reader: GgufSource): Any {
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

  private fun readMetadataValueType(reader: GgufSource): MetadataValueType {
    val index = readInt(reader)
    return MetadataValueType.fromIndex(index)
  }

  private fun readBytes(reader: GgufSource, length: Int): ByteArray = reader.readBytes(length)
  private fun skipBytes(reader: GgufSource, length: Int) = reader.skipBytes(length)
  private fun readByte(reader: GgufSource): Byte = reader.readByte()
  private fun readBoolean(reader: GgufSource): Boolean = readByte(reader).toInt() != 0
  private fun readShort(reader: GgufSource): Short = reader.readShort()
  private fun readInt(reader: GgufSource): Int = reader.readInt()
  private fun readLong(reader: GgufSource): Long = reader.readLong()
  private fun readFloat(reader: GgufSource): Float = reader.readFloat()
  private fun readDouble(reader: GgufSource): Double = reader.readDouble()

  companion object {
    private const val GGUF_MAGIC = 0x46554747
    private const val DEFAULT_ALIGNMENT = 32 // must be a power of 2
    private const val PARSE_BUFFER_SIZE = 1 shl 20
    private val SUPPORTED_GGUF_VERSIONS = listOf(2, 3)

    fun loadTensors(
      fileChannel: FileChannel,
      tensorDataOffset: Long,
      tensorInfos: Map<String, GGUFTensorInfo>
    ): Map<String, GGMLTensorEntry> {
      val arena = Arena.global()
      val tensorData =
        fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena)
      val tensorEntries = HashMap.newHashMap<String, GGMLTensorEntry>(tensorInfos.size)
      for (entry in tensorInfos.entries) {
        val ti = entry.value
        val numberOfElements: Long = FloatTensor.numberOfElementsLong(*ti.dimensions)
        val sizeInBytes = ti.ggmlType.byteSizeFor(numberOfElements)
        val memorySegment = tensorData.asSlice(ti.offset, sizeInBytes)
        tensorEntries[ti.name] = GGMLTensorEntry(ti.ggmlType, ti.dimensions, memorySegment)
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
      Timer.log("Parse $modelLabel").use {
        fileChannel.position(0L)
        return GGUF(ChannelReader(fileChannel, PARSE_BUFFER_SIZE))
      }
    }
  }
}
