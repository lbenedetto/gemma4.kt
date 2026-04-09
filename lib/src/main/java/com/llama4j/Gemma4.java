///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+
//PREVIEW
//COMPILE_OPTIONS --add-modules=jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0
//MAIN com.llama4j.Gemma4

// Gemma 4 inference in pure Java
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects
// Related project: https://github.com/mukel/llama3.java
//
// Supports GGUF models and multiple tensor formats
// Matrix-vector kernels use Java's Vector API
// CLI modes: --chat and --instruct
//
// Run:
// jbang Gemma4.java --help
package com.llama4j;

import com.llama4j.gguf.AOT;
import com.llama4j.gguf.ModelLoader;
import com.llama4j.model.*;
import com.llama4j.sampler.CategoricalSampler;
import com.llama4j.sampler.Sampler;
import com.llama4j.sampler.ToppSampler;
import com.llama4j.tokenizer.GemmaTokenizer;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.function.IntConsumer;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static java.util.Objects.requireNonNull;

public class Gemma4 {

    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler sampler;
        if (temperature == 0.0f) {
            sampler = Sampler.ARGMAX;
        } else {
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;
            if (topp <= 0 || topp >= 1) {
                innerSampler = new CategoricalSampler(rng);
            } else {
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }
            sampler = logits -> {
                int logitsSize = Math.toIntExact(logits.size());
                logits.divideInPlace(0, logitsSize, temperature);
                logits.softmaxInPlace(0, logitsSize);
                return innerSampler.sampleToken(logits);
            };
        }
        return sampler;
    }

    private static final String ANSI_GREY  = "\033[90m";
    private static final String ANSI_RESET = "\033[0m";

    private static IntConsumer plainStreamingPrinter(GemmaTokenizer tokenizer) {
        return token -> {
            if (!tokenizer.isSpecialToken(token)) {
                System.out.print(tokenizer.decode(List.of(token)));
            }
        };
    }

    private static void onThinkingStart(PrintStream thoughtOut, boolean ansi) {
        if (ansi) {
            thoughtOut.print(ANSI_GREY);
        }
        thoughtOut.println("[Start thinking]");
    }

    private static void onThinkingEnd(PrintStream thoughtOut, boolean ansi, boolean emitted) {
        if (emitted) {
            thoughtOut.println();
        }
        thoughtOut.println("[End thinking]");
        if (ansi) {
            thoughtOut.print(ANSI_RESET);
        }
        thoughtOut.println();
    }

    static boolean supportsAnsiColors(String colorMode) {
        return switch (colorMode) {
            case "on" -> true;
            case "auto" -> {
                String noColor = System.getenv("NO_COLOR");
                if (noColor != null) {
                    yield false;
                }
                String term = System.getenv("TERM");
                yield !"dumb".equalsIgnoreCase(term);
            }
            default -> false;
        };
    }

    private static IntConsumer streamingPrinter(GemmaTokenizer tokenizer, Options options) {
        if (!options.stream()) {
            return token -> {};
        }

        Integer channelOpen = tokenizer.getSpecialTokens().get("<|channel>");
        Integer channelClose = tokenizer.getSpecialTokens().get("<channel|>");
        if (channelOpen == null || channelClose == null) {
            return plainStreamingPrinter(tokenizer);
        }

        boolean thinkEnabled = options.think();
        PrintStream thoughtOut = options.thinkInline() ? System.out : System.err;
        boolean ansi = options.colors();
        boolean[] inChannel = {false};
        boolean[] emitted = {false};
        return token -> {
            if (token == channelOpen) {
                if (thinkEnabled) {
                    onThinkingStart(thoughtOut, ansi);
                }
                inChannel[0] = true;
                emitted[0] = false;
                return;
            }
            if (token == channelClose) {
                if (thinkEnabled) {
                    onThinkingEnd(thoughtOut, ansi, emitted[0]);
                }
                inChannel[0] = false;
                emitted[0] = false;
                return;
            }
            if (!tokenizer.isSpecialToken(token)) {
                String text = tokenizer.decode(List.of(token));
                if (inChannel[0]) {
                    if (thinkEnabled) {
                        thoughtOut.print(text);
                        emitted[0] = true;
                    }
                } else {
                    System.out.print(text);
                }
            }
        };
    }

    private static List<Integer> visibleTokens(GemmaTokenizer tokenizer, List<Integer> tokens, boolean think) {
        return think ? stripThoughtChannelTokens(tokenizer, tokens) : tokens;
    }

    private static List<Integer> stripThoughtChannelTokens(GemmaTokenizer tokenizer, List<Integer> tokens) {
        Integer channelOpen = tokenizer.getSpecialTokens().get("<|channel>");
        Integer channelClose = tokenizer.getSpecialTokens().get("<channel|>");
        if (channelOpen == null || channelClose == null || tokens.isEmpty()) {
            return tokens;
        }
        List<Integer> out = new ArrayList<>(tokens.size());
        boolean inChannel = false;
        for (int tok : tokens) {
            if (tok == channelOpen) { inChannel = true; continue; }
            if (tok == channelClose) { inChannel = false; continue; }
            if (!inChannel) { out.add(tok); }
        }
        return out;
    }

    static void runInteractive(Llama model, Sampler sampler, Options options) {
        LlamaState state = null;
        GemmaChatFormat chatFormat = new GemmaChatFormat(model.tokenizer());
        List<Integer> conversationTokens = new ArrayList<>();
        if (options.think()) {
            conversationTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt()));
        } else if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new Message(Role.SYSTEM, options.systemPrompt())));
        }
        int startPosition = 0;
        Scanner in = new Scanner(System.in);
        loop: while (true) {
            System.out.print("> ");
            System.out.flush();
            String userText = in.nextLine();
            switch (userText) {
                case "/quit":
                case "/exit": break loop;
                case "/context": {
                    System.out.printf("%d out of %d context tokens used (%d tokens remaining)%n",
                            conversationTokens.size(),
                            options.maxTokens(),
                            options.maxTokens() - conversationTokens.size());
                    continue;
                }
            }
            if (state == null) {
                state = model.createNewState();
            }
            conversationTokens.addAll(chatFormat.encodeMessage(new Message(Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeHeader(new Message(Role.MODEL, "")));

            Set<Integer> stopTokens = chatFormat.getStopTokens();
            IntConsumer printer = streamingPrinter(model.tokenizer(), options);
            List<Integer> responseTokens = Llama.generateTokens(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), options.colors(), printer);
            Integer stopToken = null;
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                stopToken = responseTokens.getLast();
                responseTokens.removeLast();
            }
            List<Integer> visibleResponseTokens = visibleTokens(model.tokenizer(), responseTokens, options.think());
            conversationTokens.addAll(responseTokens);
            if (stopToken != null) {
                conversationTokens.add(stopToken);
            }
            startPosition = conversationTokens.size();
            if (!options.stream()) {
                String responseText = model.tokenizer().decode(visibleResponseTokens);
                System.out.println(responseText);
            }
            if (stopToken == null) {
                System.err.println("Ran out of context length...");
                break;
            }
        }
    }

    static void runInstructOnce(Llama model, Sampler sampler, Options options) {
        var prompt = requireNonNull(options.prompt());
        LlamaState state = model.createNewState();
        GemmaChatFormat chatFormat = new GemmaChatFormat(model.tokenizer());
        List<Integer> promptTokens = new ArrayList<>();

        if (options.suffix() != null) {
            promptTokens.addAll(chatFormat.encodeFillInTheMiddle(prompt, options.suffix()));
        } else {
            if (options.think()) {
                promptTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt()));
            } else if (options.systemPrompt() != null) {
                promptTokens.addAll(chatFormat.encodeMessage(new Message(Role.SYSTEM, options.systemPrompt())));
            }
            promptTokens.addAll(chatFormat.encodeMessage(new Message(Role.USER, prompt)));
            promptTokens.addAll(chatFormat.encodeHeader(new Message(Role.MODEL, "")));
        }

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        IntConsumer printer = streamingPrinter(model.tokenizer(), options);
        List<Integer> responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), options.colors(), printer);
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        List<Integer> visibleResponseTokens = visibleTokens(model.tokenizer(), responseTokens, options.think());
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(visibleResponseTokens);
            System.out.println(responseText);
        }
    }

    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Llama model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        if (model == null) {
            model = ModelLoader.loadModel(options.modelPath(), options.maxTokens());
        }
        Sampler sampler = selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
        if (options.interactive()) {
            runInteractive(model, sampler, options);
        } else {
            runInstructOnce(model, sampler, options);
        }
    }
}
