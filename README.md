# gemma4.kt

Gemma 4 inference in pure Kotlin/Java. Supports GGUF models with quantized and
dense tensor formats. Matrix-vector kernels use Java's Vector API.

## Requirements

- Java 25+ (Vector API via `--add-modules=jdk.incubator.vector`)
- A Gemma 4 GGUF model file

## Library API

Add the `lib` module as a dependency. The entry point is `GemmaModel`.

### Loading a model

```kotlin
val model = GemmaModel.load(Path.of("gemma-4-E2B-it-Q8_0.gguf"))
```

An optional `contextLength` parameter overrides the model's built-in context window:

```kotlin
val model = GemmaModel.load(Path.of("gemma-4-E2B-it-Q8_0.gguf"), contextLength = 8192)
```

### Single-turn generation

```kotlin
val result = model.generate("Why is the sky blue?")
println(result.text)
```

### Generation options

All options are set via a configuration DSL block:

```kotlin
val result = model.generate("Write a haiku") {
    temperature = 0.7f   // 0 = greedy, higher = more random (default: 1.0)
    topP = 0.9f          // nucleus sampling threshold (default: 0.95)
    maxTokens = 256      // default: 1024
    systemPrompt = "You are a creative writing assistant."
    seed = 42L           // for reproducibility
}
```

### Streaming

Set `onToken` to receive decoded text pieces as they are generated:

```kotlin
model.generate("Tell me a joke") {
    temperature = 0.8f
    onToken = { piece -> print(piece) }
}
```

`onToken` is called during generation; `result.text` always contains the complete
response regardless of whether streaming is used.

### Thinking mode

When enabled, the model reasons internally before answering. Thinking tokens are
excluded from `result.text` and the `onToken` stream, but available separately:

```kotlin
val result = model.generate("What is 17 × 34?") {
    thinking = true
}
println(result.text)     // final answer
println(result.thinking) // internal reasoning (null if model doesn't support it)
```

### Multi-turn chat

`chat()` returns a `ChatSession` that retains conversation history and `LlamaState`
across turns:

```kotlin
val chat = model.chat {
    systemPrompt = "You are a helpful assistant."
    temperature = 0.9f
}

println(chat.send("Hello!").text)
println(chat.send("What did I just say?").text)  // model has full prior context
```

Check context window usage:

```kotlin
println("${chat.contextUsed} / ${chat.contextUsed + chat.contextRemaining} tokens used")
```

Call `reset()` to clear history while keeping the configuration and system prompt:

```kotlin
chat.reset()
```

### Fill-in-the-middle

For code completion with a known suffix (requires a FIM-capable model):

```kotlin
val result = model.fillInMiddle(
    prefix = "fun greet(name: String) = ",
    suffix = ""
)
println(result.text)
```

## Command-line interface

The `cli` module provides a CLI:

```
./gradlew :cli:run --args="--model gemma-4-E2B-it-Q8_0.gguf --prompt \"Tell me a joke\""
```

### Options

| Option | Description |
|---|---|
| `--model`, `-m` | Path to `.gguf` file (required) |
| `--prompt`, `-p` | Input prompt (required in instruct mode) |
| `--interactive`, `--chat`, `-i` | Run in interactive chat mode |
| `--system-prompt`, `-sp` | System prompt |
| `--temperature` | Sampling temperature (default: 1.0) |
| `--top-p` | Nucleus sampling threshold (default: 0.95) |
| `--seed` | Random seed |
| `--max-tokens`, `-n` | Max tokens to generate (default: 1024) |
| `--stream` | Stream tokens to stdout (default: true) |
| `--think` | Thinking mode: `off`, `on` (stderr), `inline` (stdout) |
| `--color` | Colorize thinking output: `on`, `off`, `auto` (default) |

### Interactive commands

| Command | Description |
|---|---|
| `/quit`, `/exit` | Exit |
| `/context` | Show context token usage |
