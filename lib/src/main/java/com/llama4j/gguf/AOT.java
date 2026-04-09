package com.llama4j.gguf;

import com.llama4j.Options;
import com.llama4j.model.Llama;
import com.llama4j.model.LlamaConfiguration;
import com.llama4j.model.LlamaWeights;
import com.llama4j.model.RoPE;
import com.llama4j.util.Timer;
import kotlin.Pair;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;

public final class AOT {

    private static final PartialModel PRELOADED_GGUF = preLoadGGUF(System.getProperty("gemma4.PreloadGGUF"));

    private static PartialModel preLoadGGUF(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            return null;
        }
        try {
            Path path = Path.of(modelPath);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                throw new IllegalArgumentException("Cannot pre-load model: " + path);
            }
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                GGUF gguf = GGUF.loadModel(fileChannel, path.toString());
                Llama base = ModelLoader.loadModel(null, gguf, Options.DEFAULT_MAX_TOKENS, false);
                // Precompute RoPE frequencies at build time (pure Java arrays, survives native-image)
                LlamaConfiguration config = base.configuration();
                Pair<float[], float[]> ropeFreqsSWA = RoPE.precomputeFreqsCis(
                        config.contextLength, config.headSizeSWA, config.ropeThetaSWA);
                // Read rope_freqs from model file
                Pair<float[], float[]> ropeFreqsFull;
                Map<String, GGMLTensorEntry> tmpEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                FloatBuffer ropeFreqsBuf = ModelLoader.toFloatBuffer(tmpEntries.get("rope_freqs.weight"));
                float[] modelRopeFreqs = new float[ropeFreqsBuf.remaining()];
                ropeFreqsBuf.get(modelRopeFreqs);
                ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(
                        config.contextLength, config.headSizeFull, config.ropeTheta, modelRopeFreqs);
                return new PartialModel(
                        path.getFileName().toString(), base,
                        gguf.getTensorDataOffset(), gguf.getTensorInfos(),
                        ropeFreqsSWA, ropeFreqsFull);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static Llama tryUsePreLoaded(Path modelPath, int contextLength) throws IOException {
        PartialModel preLoaded = AOT.PRELOADED_GGUF;
        if (preLoaded == null) {
            return null;
        }
        String optionsModel = modelPath.getFileName().toString();
        String preLoadedModel = preLoaded.modelFileName();
        if (!Objects.equals(optionsModel, preLoadedModel)) {
            return null;
        }
        Llama baseModel = preLoaded.model();
        try (var timer = Timer.log("Load tensors from pre-loaded model");
             var fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, preLoaded.tensorDataOffset(), preLoaded.tensorInfos());
            LlamaWeights weights = ModelLoader.loadWeightsWithRoPE(tensorEntries, baseModel.configuration(),
                    preLoaded.ropeFreqsSWA(), preLoaded.ropeFreqsFull());
            return new Llama(baseModel.configuration().withContextLength(contextLength), baseModel.tokenizer(), weights);
        }
    }
}
