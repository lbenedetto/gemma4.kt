package com.llama4j;

import org.jspecify.annotations.Nullable;

import java.io.PrintStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

import static java.util.Objects.requireNonNull;

public record Options(
        Path modelPath,
        @Nullable String prompt,
        @Nullable String suffix,
        @Nullable String systemPrompt,
        boolean interactive,
        float temperature,
        float topp,
        long seed,
        int maxTokens,
        boolean stream,
        boolean echo,
        boolean think,
        boolean thinkInline,
        boolean colors
) {

  public static final int DEFAULT_MAX_TOKENS = 1024;

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
                default ->
                    require(false, "Invalid argument for %s: expected off|on|inline (or false|true|stdout), got %s", optionName, nextArg);
              }
            }
            default -> require(false, "Unknown option: %s", optionName);
          }
        }
      }
    }
    require(List.of("on", "off", "auto").contains(colorMode), "Invalid argument: --color must be one of on|off|auto");
    boolean color = Gemma4.supportsAnsiColors(colorMode);
    require(modelPath != null, "Missing argument: --model <path> is required");
    requireNonNull(modelPath);
    require(interactive || prompt != null, "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
    require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
    require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
    return new Options(modelPath, prompt, suffix, systemPrompt, interactive, temperature, topp, seed, maxTokens, stream, echo, think, thinkInline, color);
  }
}
