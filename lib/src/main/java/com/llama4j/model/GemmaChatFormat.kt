package com.llama4j.model;

import com.llama4j.tokenizer.GemmaTokenizer;
import org.jspecify.annotations.Nullable;

import java.util.*;

import static java.util.Objects.requireNonNull;

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
        this.beginOfSentence = requireNonNull(specialTokens.get("<bos>"));
        this.startOfTurn = requireNonNull(specialTokens.get("<|turn>"));
        this.endOfTurn = requireNonNull(specialTokens.get("<turn|>"));
        this.endOfSentence = requireNonNull(specialTokens.get("<eos>"));

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

    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startOfTurn);
        tokens.addAll(tokenizer.encode(message.role().toString()));
        tokens.addAll(this.tokenizer.encode("\n"));
        return tokens;
    }

    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encode(message.content().strip()));
        tokens.add(endOfTurn);
        tokens.addAll(this.tokenizer.encode("\n"));
        return tokens;
    }

    public List<Integer> encodeSystemThinkingTurn(@Nullable String systemPrompt) {
        // Matches Gemma4 template with enable_thinking=true:
        // <|turn>system\n<|think|>[system_content]<turn|>\n
        List<Integer> tokens = new ArrayList<>(encodeHeader(new Message(Role.SYSTEM, "")));
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



    public List<Integer> encodeFillInTheMiddle(String prefix, String suffix) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(this.fimPrefix);
        tokens.addAll(tokenizer.encode(prefix));
        tokens.add(this.fimSuffix);
        tokens.addAll(tokenizer.encode(suffix));
        tokens.add(this.fimMiddle);
        return tokens;
    }


}
