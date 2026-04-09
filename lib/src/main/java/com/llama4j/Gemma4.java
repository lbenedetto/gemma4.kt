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

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.*;
import java.util.function.IntConsumer;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

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
            case "off" -> false;
            case "auto" -> {
                if (System.console() == null) {
                    yield false;
                }
                String noColor = System.getenv("NO_COLOR");
                if (noColor != null) {
                    yield false;
                }
                String term = System.getenv("TERM");
                yield term == null || !"dumb".equalsIgnoreCase(term);
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
        Llama.State state = null;
        GemmaChatFormat chatFormat = new GemmaChatFormat(model.tokenizer());
        List<Integer> conversationTokens = new ArrayList<>();
        if (options.think()) {
            conversationTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt()));
        } else if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new GemmaChatFormat.Message(GemmaChatFormat.Role.SYSTEM, options.systemPrompt())));
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
            conversationTokens.addAll(chatFormat.encodeMessage(new GemmaChatFormat.Message(GemmaChatFormat.Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeHeader(new GemmaChatFormat.Message(GemmaChatFormat.Role.MODEL, "")));

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
        Llama.State state = model.createNewState();
        GemmaChatFormat chatFormat = new GemmaChatFormat(model.tokenizer());
        List<Integer> promptTokens = new ArrayList<>();

        if (options.suffix() != null) {
            promptTokens.addAll(chatFormat.encodeFillInTheMiddle(options.prompt(), options.suffix()));
        } else {
            if (options.think()) {
                promptTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt()));
            } else if (options.systemPrompt() != null) {
                promptTokens.addAll(chatFormat.encodeMessage(new GemmaChatFormat.Message(GemmaChatFormat.Role.SYSTEM, options.systemPrompt())));
            }
            promptTokens.addAll(chatFormat.encodeMessage(new GemmaChatFormat.Message(GemmaChatFormat.Role.USER, options.prompt())));
            promptTokens.addAll(chatFormat.encodeHeader(new GemmaChatFormat.Message(GemmaChatFormat.Role.MODEL, "")));
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

    static final int DEFAULT_MAX_TOKENS = 1024;

    record Options(Path modelPath, String prompt, String suffix, String systemPrompt, boolean interactive,
                   float temperature, float topp, long seed, int maxTokens, boolean stream, boolean echo,
                   boolean think, boolean thinkInline, boolean colors) {

        Options {
            require(modelPath != null, "Missing argument: --model <path> is required");
            require(interactive || prompt != null, "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
            require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
            require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
        }

        static void require(boolean condition, String messageFormat, Object... args) {
            if (!condition) {
                System.out.println("ERROR " + messageFormat.formatted(args));
                System.out.println();
                printUsage(System.out);
                System.exit(-1);
            }
        }

        static boolean parseBooleanOption(String optionName, String value) {
            return switch (value.toLowerCase(Locale.ROOT)) {
                case "true", "on" -> true;
                case "false", "off" -> false;
                default -> {
                    require(false, "Invalid argument for %s: expected true|false|on|off, got %s", optionName, value);
                    yield false;
                }
            };
        }

        static void printUsage(PrintStream out) {
            out.println("Usage:  jbang Gemma4.java [options]");
            out.println();
            out.println("Options:");
            out.println("  --model, -m <path>            required, path to .gguf file");
            out.println("  --interactive, --chat, -i     run in chat mode");
            out.println("  --instruct                    run in instruct (once) mode, default mode");
            out.println("  --prompt, -p <string>         input prompt");
            out.println("  --suffix <string>             suffix for fill-in-the-middle request");
            out.println("  --system-prompt, -sp <string> system prompt for chat/instruct mode");
            out.println("  --temperature, -temp <float>  temperature in [0,inf], default 1.0");
            out.println("  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95");
            out.println("  --seed <long>                 random seed, default System.nanoTime()");
            out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default " + DEFAULT_MAX_TOKENS);
            out.println("  --stream <boolean>            print tokens during generation; accepts true|false|on|off, default true");
            out.println("  --echo <boolean>              print ALL tokens to stderr; accepts true|false|on|off, default false");
            out.println("  --color <on|off|auto>         colorize thinking output in terminal (default: auto)");
            out.println("  --think <off|on|inline>       off: disable thoughts, on: thoughts to stderr, inline: thoughts to stdout");
            out.println();
            out.println("Interactive commands:");
            out.println("  /quit, /exit                  exit the chat");
            out.println("  /context                      show context token usage");
            out.println();
            out.println("Examples:");
            out.println("  jbang Gemma4.java --model gemma-4-E2B-it-Q8_0.gguf --chat");
            out.println("  jbang Gemma4.java --model gemma-4-E2B-it-Q8_0.gguf --prompt \"Tell me a joke\"");
            out.println("  jbang Gemma4.java --model gemma-4-E2B-it-Q8_0.gguf --chat --system-prompt \"You are a helpful assistant\"");
        }

        static Options parseOptions(String[] args) {
            String prompt = null;
            String suffix = null;
            String systemPrompt = null;
            float temperature = 1f;
            float topp = 0.95f;
            Path modelPath = null;
            long seed = System.nanoTime();
            int maxTokens = DEFAULT_MAX_TOKENS;
            boolean interactive = false;
            boolean stream = true;
            boolean echo = false;
            boolean think = false;
            boolean thinkInline = false;
            String colorMode = "auto";

            for (int i = 0; i < args.length; i++) {
                String optionName = args[i];
                require(optionName.startsWith("-"), "Invalid option %s", optionName);
                switch (optionName) {
                    case "--interactive", "--chat", "-i" -> interactive = true;
                    case "--instruct" -> interactive = false;
                    case "--help", "-h" -> {
                        printUsage(System.out);
                        System.exit(0);
                    }
                    default -> {
                        String nextArg;
                        if (optionName.contains("=")) {
                            String[] parts = optionName.split("=", 2);
                            optionName = parts[0];
                            nextArg = parts[1];
                        } else {
                            require(i + 1 < args.length, "Missing argument for option %s", optionName);
                            nextArg = args[i + 1];
                            i += 1;
                        }
                        switch (optionName) {
                            case "--prompt", "-p" -> prompt = nextArg;
                            case "--suffix" -> suffix = nextArg;
                            case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                            case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                            case "--top-p" -> topp = Float.parseFloat(nextArg);
                            case "--model", "-m" -> modelPath = Path.of(nextArg);
                            case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                            case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                            case "--stream" -> stream = parseBooleanOption(optionName, nextArg);
                            case "--echo" -> echo = parseBooleanOption(optionName, nextArg);
                            case "--color" -> colorMode = nextArg.toLowerCase(Locale.ROOT);
                            case "--think" -> {
                                String thinkMode = nextArg.toLowerCase(Locale.ROOT);
                                thinkInline = List.of("inline", "stdout").contains(thinkMode);
                                switch (thinkMode) {
                                    case "on", "true", "inline", "stdout" -> think = true;
                                    case "off", "false" -> think = false;
                                    default -> require(false, "Invalid argument for %s: expected off|on|inline (or false|true|stdout), got %s", optionName, nextArg);
                                }
                            }
                            default -> require(false, "Unknown option: %s", optionName);
                        }
                    }
                }
            }
            require(List.of("on", "off", "auto").contains(colorMode), "Invalid argument: --color must be one of on|off|auto");
            boolean color = Gemma4.supportsAnsiColors(colorMode);
            return new Options(modelPath, prompt, suffix, systemPrompt, interactive, temperature, topp, seed, maxTokens, stream, echo, think, thinkInline, color);
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
