package com.llama4j.model;

import com.llama4j.tokenizer.GemmaTokenizer;

import java.util.*;

public class GemmaChatFormat {

    protected final GemmaTokenizer tokenizer;
    protected final int beginOfSentence;
    protected final int startOfTurn;
    protected final int endOfTurn;
    protected final int endOfSentence;
    protected final int fimSuffix;
    protected final int fimPrefix;
    protected final int fimMiddle;
    protected final int fileSeparator;

    public GemmaChatFormat(GemmaTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.beginOfSentence = specialTokens.get("<bos>");
        this.startOfTurn = specialTokens.get("<|turn>");
        this.endOfTurn = specialTokens.get("<turn|>");
        this.endOfSentence = specialTokens.get("<eos>");

        this.fimSuffix = specialTokens.getOrDefault("<|fim_suffix|>", -1);
        this.fimPrefix = specialTokens.getOrDefault("<|fim_prefix|>", -1);
        this.fimMiddle = specialTokens.getOrDefault("<|fim_middle|>", -1);
        this.fileSeparator = specialTokens.getOrDefault("<|file_separator|>", -1);
    }

    public Set<Integer> getStopTokens() {
        Set<Integer> tokens = new HashSet<>();
        tokens.add(endOfSentence);
        tokens.add(endOfTurn);
        if (fimSuffix != -1) {
            tokens.add(fimSuffix);
        }
        if (fimPrefix != -1) {
            tokens.add(fimPrefix);
        }
        if (fimMiddle != -1) {
            tokens.add(fimMiddle);
        }
        if (fileSeparator != -1) {
            tokens.add(fileSeparator);
        }
        return tokens;
    }

    public List<Integer> encodeHeader(GemmaChatFormat.Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startOfTurn);
        tokens.addAll(tokenizer.encode(message.role().toString()));
        tokens.addAll(this.tokenizer.encode("\n"));
        return tokens;
    }

    public List<Integer> encodeMessage(GemmaChatFormat.Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encode(message.content().strip()));
        tokens.add(endOfTurn);
        tokens.addAll(this.tokenizer.encode("\n"));
        return tokens;
    }

    public List<Integer> encodeSystemThinkingTurn(String systemPrompt) {
        // Matches Gemma4 template with enable_thinking=true:
        // <|turn>system\n<|think|>[system_content]<turn|>\n
        List<Integer> tokens = new ArrayList<>();
        tokens.addAll(encodeHeader(new Message(Role.SYSTEM, "")));
        Integer thinkToken = tokenizer.getSpecialTokens().get("<|think|>");
        if (thinkToken != null) {
            tokens.add(thinkToken);
        }
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            tokens.addAll(tokenizer.encode(systemPrompt.trim()));
        }
        tokens.add(endOfTurn);
        tokens.addAll(tokenizer.encode("\n"));
        return tokens;
    }

    public record Message(GemmaChatFormat.Role role, String content) {
    }

    public List<Integer> encodeFillInTheMiddle(String prefix, String suffix) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(this.fimPrefix);
        tokens.addAll(tokenizer.encode(prefix));
        tokens.add(this.fimSuffix);
        tokens.addAll(tokenizer.encode(suffix));
        tokens.add(this.fimMiddle);
        return tokens;
    }

    public record Role(String name) {
        public static GemmaChatFormat.Role SYSTEM = new GemmaChatFormat.Role("system");
        public static GemmaChatFormat.Role USER = new GemmaChatFormat.Role("user");
        public static GemmaChatFormat.Role MODEL = new GemmaChatFormat.Role("model");

        @Override
        public String toString() {
            return name;
        }
    }
}
