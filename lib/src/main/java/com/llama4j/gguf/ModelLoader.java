package com.llama4j.gguf;

import com.llama4j.floattensor.*;
import com.llama4j.model.Llama;
import com.llama4j.model.RoPE;
import com.llama4j.tokenizer.GemmaTokenizer;
import com.llama4j.tokenizer.Vocabulary;
import com.llama4j.util.Pair;
import com.llama4j.util.Timer;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.function.IntFunction;

public final class ModelLoader {

    private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        return new Vocabulary(tokens, scores);
    }

    public static Llama loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load Gemma4 model")) {
            try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
                GGUF gguf = GGUF.loadModel(fileChannel, ggufPath.toString());
                return loadModel(fileChannel, gguf, contextLength, true);
            }
        }
    }

    public static Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        Map<String, Object> metadata = gguf.getMetadata();

        Vocabulary vocabulary = loadVocabulary(metadata);
        GemmaTokenizer tokenizer = createTokenizer(metadata, vocabulary);

        int modelContextLength = (int) metadata.get("gemma4.context_length");
        if (contextLength < 0 || modelContextLength < contextLength) {
            contextLength = modelContextLength;
        }

        int embeddingLength = (int) metadata.get("gemma4.embedding_length");
        int numberOfHeads = (int) metadata.get("gemma4.attention.head_count");
        int numberOfLayers = (int) metadata.get("gemma4.block_count");

        int headSizeFull = (int) metadata.get("gemma4.attention.key_length");
        int headSizeSWA = (int) metadata.get("gemma4.attention.key_length_swa");
        int slidingWindow = (int) metadata.get("gemma4.attention.sliding_window");
        float logitSoftcapping = (float) metadata.getOrDefault("gemma4.final_logit_softcapping", 0f);
        float rmsNormEps = (float) metadata.getOrDefault("gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeTheta = (float) metadata.getOrDefault("gemma4.rope.freq_base", 1000000f);
        float ropeThetaSWA = (float) metadata.getOrDefault("gemma4.rope.freq_base_swa", 10000f);

        // MoE parameters
        int expertCount = (int) metadata.getOrDefault("gemma4.expert_count", 0);
        int expertUsedCount = (int) metadata.getOrDefault("gemma4.expert_used_count", 0);
        int expertFeedForwardLength = (int) metadata.getOrDefault("gemma4.expert_feed_forward_length", 0);

        // Per-layer feed forward lengths (scalar for uniform, array for variable)
        int[] feedForwardLength;
        Object ffnRaw = metadata.get("gemma4.feed_forward_length");
        if (ffnRaw instanceof int[] arr) {
            feedForwardLength = arr;
        } else {
            feedForwardLength = new int[numberOfLayers];
            Arrays.fill(feedForwardLength, (int) ffnRaw);
        }

        Map<String, GGUF.GGUFTensorInfo> tensorInfos = gguf.getTensorInfos();

        // Derive isSWA per layer from Q norm weight size (256 = SWA, 512 = full attention)
        boolean[] isSWA;
        Object swaRaw = metadata.get("gemma4.attention.sliding_window_pattern");
        if (swaRaw instanceof boolean[] arr) {
            isSWA = arr;
        } else {
            // Derive from tensor shapes: check Q norm size per layer
            isSWA = new boolean[numberOfLayers];
            for (int i = 0; i < numberOfLayers; i++) {
                GGUF.GGUFTensorInfo qNorm = tensorInfos.get("blk." + i + ".attn_q_norm.weight");
                if (qNorm != null) {
                    long qNormSize = FloatTensor.numberOfElementsLong(qNorm.dimensions());
                    isSWA[i] = (qNormSize == headSizeSWA);
                } else {
                    isSWA[i] = (i % 5 != 4); // fallback
                }
            }
        }

        // Derive per-layer KV head count from K weight shapes
        int[] numberOfKeyValueHeadsPerLayer = new int[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            GGUF.GGUFTensorInfo kWeight = tensorInfos.get("blk." + i + ".attn_k.weight");
            int headSize = isSWA[i] ? headSizeSWA : headSizeFull;
            if (kWeight != null) {
                long kDim = kWeight.dimensions()[1];
                numberOfKeyValueHeadsPerLayer[i] = (int) (kDim / headSize);
            } else {
                numberOfKeyValueHeadsPerLayer[i] = numberOfHeads; // fallback
            }
        }

        // Shared KV layers: last N layers reuse KV from earlier layers
        int sharedKvLayers = (int) metadata.getOrDefault("gemma4.attention.shared_kv_layers", 0);
        int nLayerKvFromStart = numberOfLayers - sharedKvLayers;

        int embeddingLengthPerLayer = (int) metadata.getOrDefault("gemma4.embedding_length_per_layer_input", 0);

        Llama.Configuration config = new Llama.Configuration(
                embeddingLength,
                feedForwardLength,
                numberOfLayers,
                numberOfHeads,
                numberOfKeyValueHeadsPerLayer,
                vocabulary.size(),
                contextLength,
                rmsNormEps,
                ropeTheta,
                ropeThetaSWA,
                headSizeFull,
                headSizeSWA,
                slidingWindow,
                logitSoftcapping,
                isSWA,
                nLayerKvFromStart,
                embeddingLengthPerLayer,
                expertCount,
                expertUsedCount,
                expertFeedForwardLength
        );

        if (!loadWeightsFlag) {
            return new Llama(config, tokenizer, null);
        }

        Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), tensorInfos);
        Llama.Weights qw = loadWeights(tensorEntries, config);
        return new Llama(config, tokenizer, qw);
    }

    public static Llama.Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config) {
        Pair<float[], float[]> ropeFreqsSWA = RoPE.precomputeFreqsCis(config.contextLength, config.headSizeSWA, config.ropeThetaSWA);
        FloatBuffer ropeFreqsBuf = toFloatBuffer(tensorEntries.get("rope_freqs.weight"));
        float[] modelRopeFreqs = new float[ropeFreqsBuf.remaining()];
        ropeFreqsBuf.get(modelRopeFreqs);
        Pair<float[], float[]> ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(config.contextLength, config.headSizeFull, config.ropeTheta, modelRopeFreqs);
        return loadWeightsWithRoPE(tensorEntries, config, ropeFreqsSWA, ropeFreqsFull);
    }

    public static Llama.Weights loadWeightsWithRoPE(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config,
                                                     Pair<float[], float[]> ropeFreqsSWA, Pair<float[], float[]> ropeFreqsFull) {
        int numberOfLayers = config.numberOfLayers;

        FloatTensor tokenEmbeddingTable = loadQuantized(tensorEntries.get("token_embd.weight"));

        // Load per-layer output scale (scalar per layer)
        float[] layerOutputScale = new float[config.numberOfLayers];
        for (int i = 0; i < config.numberOfLayers; i++) {
            GGMLTensorEntry scaleEntry = tensorEntries.get("blk." + i + ".layer_output_scale.weight");
            if (scaleEntry != null) {
                layerOutputScale[i] = toFloatBuffer(scaleEntry).get(0);
            } else {
                layerOutputScale[i] = 1.0f;
            }
        }

        // Load per-layer embedding weights (if present)
        FloatTensor perLayerTokenEmbd = null;
        FloatTensor perLayerModelProj = null;
        FloatBuffer perLayerProjNorm = null;
        FloatTensor[] perLayerInpGate = null;
        FloatTensor[] perLayerProj = null;
        FloatBuffer[] perLayerPostNorm = null;

        if (config.embeddingLengthPerLayer > 0 && tensorEntries.containsKey("per_layer_token_embd.weight")) {
            perLayerTokenEmbd = loadQuantized(tensorEntries.get("per_layer_token_embd.weight"));
            perLayerModelProj = loadQuantized(tensorEntries.get("per_layer_model_proj.weight"));
            perLayerProjNorm = toFloatBuffer(tensorEntries.get("per_layer_proj_norm.weight"));
            perLayerInpGate = loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".inp_gate.weight"));
            perLayerProj = loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".proj.weight"));
            perLayerPostNorm = loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".post_norm.weight"));
        }

        // Load V weights (nullable: layers without V use K as V)
        FloatTensor[] wv = new FloatTensor[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            GGMLTensorEntry vEntry = tensorEntries.get("blk." + i + ".attn_v.weight");
            wv[i] = vEntry != null ? loadQuantized(vEntry) : null;
        }

        // Load MoE weights (if present)
        FloatTensor[] ffnGateInp = null;
        FloatBuffer[] ffnGateInpScale = null;
        FloatTensor[] ffnGateUpExps = null;
        FloatTensor[] ffnDownExps = null;
        FloatBuffer[] ffnDownExpsScale = null;
        FloatBuffer[] ffnPostNorm1 = null;
        FloatBuffer[] preFfwNorm2 = null;
        FloatBuffer[] ffnPostNorm2 = null;

        if (config.isMoE()) {
            ffnGateInp = loadArrayOfQuantized(numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp.weight"));
            ffnGateInpScale = loadArrayOfFloatBuffer(numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp.scale"));
            ffnGateUpExps = loadArrayOfQuantized(numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate_up_exps.weight"));
            ffnDownExps = loadArrayOfQuantized(numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down_exps.weight"));
            ffnDownExpsScale = loadArrayOfFloatBuffer(numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down_exps.scale"));
            ffnPostNorm1 = loadArrayOfFloatBuffer(numberOfLayers, i -> tensorEntries.get("blk." + i + ".post_ffw_norm_1.weight"));
            preFfwNorm2 = loadArrayOfFloatBuffer(numberOfLayers, i -> tensorEntries.get("blk." + i + ".pre_ffw_norm_2.weight"));
            ffnPostNorm2 = loadArrayOfFloatBuffer(numberOfLayers, i -> tensorEntries.get("blk." + i + ".post_ffw_norm_2.weight"));
        }

        return new Llama.Weights(
                tokenEmbeddingTable,
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                wv,
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),
                toFloatBuffer(tensorEntries.get("output_norm.weight")),
                layerOutputScale,
                FloatBuffer.wrap(ropeFreqsFull.first()),
                FloatBuffer.wrap(ropeFreqsFull.second()),
                FloatBuffer.wrap(ropeFreqsSWA.first()),
                FloatBuffer.wrap(ropeFreqsSWA.second()),
                tensorEntries.containsKey("output.weight")
                        ? loadQuantized(tensorEntries.get("output.weight"))
                        : tokenEmbeddingTable,
                perLayerTokenEmbd, perLayerModelProj, perLayerProjNorm,
                perLayerInpGate, perLayerProj, perLayerPostNorm,
                ffnGateInp, ffnGateInpScale, ffnGateUpExps, ffnDownExps, ffnDownExpsScale,
                ffnPostNorm1, preFfwNorm2, ffnPostNorm2
        );
    }

    private static GemmaTokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        return new GemmaTokenizer(vocabulary, tokenTypes);
    }

    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        return FloatTensorFactory.create(ggmlType, entry);
    }

    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }
}
