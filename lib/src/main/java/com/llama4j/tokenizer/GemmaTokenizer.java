package com.llama4j.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GemmaTokenizer {
    private final Vocabulary vocabulary;
    private final Map<String, Integer> specialTokens;
    private final int[] tokenType;
    private final int byte0;

    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    public boolean isSpecialToken(int tokenIndex) {
        return tokenType[tokenIndex] != 1;
    }

    public GemmaTokenizer(Vocabulary vocabulary, int[] tokenType) {
        this.vocabulary = vocabulary;
        this.tokenType = tokenType.clone();
        int endOfTurn = vocabulary.getIndex("<turn|>").orElseThrow();
        for (int i = 0; i <= endOfTurn; ++i) {
            if (this.tokenType[i] == 1) {
                this.tokenType[i] = 6;
            }
        }
        this.byte0 = vocabulary.getIndex("<0x00>").orElseThrow();
        this.specialTokens = buildSpecialTokens(this.tokenType)
                .stream()
                .collect(Collectors.toMap(t -> vocabulary.get(t), t -> t));
    }

    private static List<Integer> buildSpecialTokens(int[] tokenType) {
        return IntStream.range(0, tokenType.length)
                .filter(t -> tokenType[t] != 1)
                .boxed()
                .toList();
    }

    public List<Integer> encode(String text) {
        return encodeImpl(text.replace(' ', '\u2581'));
    }

    private List<Integer> encodeImpl(String text) {
        List<Integer> tokens = new ArrayList<>();

        for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
            cpi = text.codePointAt(i);

            String singleCodepoint = Character.toString(cpi);
            int id = vocabulary.getIndex(singleCodepoint).orElse(-1);

            if (id != -1) {
                tokens.add(id);
            } else {
                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
                    tokens.add(Byte.toUnsignedInt(b) + byte0);
                }
            }
        }

        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < tokens.size() - 1; ++i) {
                String str_buffer = vocabulary.get(tokens.get(i)) + vocabulary.get(tokens.get(i + 1));
                int id = vocabulary.getIndex(str_buffer).orElse(-1);
                if (id != -1 && vocabulary.getScore(id) > best_score) {
                    best_score = vocabulary.getScore(id);
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break;
            }

            tokens.set(best_idx, best_id);
            tokens.remove(best_idx + 1);
        }

        return tokens;
    }

    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            if (isSpecialToken(token)) {
                String prefix = "<0x";
                String suffix = ">";
                if (tokenString.length() == 6 && tokenString.startsWith(prefix) && tokenString.endsWith(suffix)) {
                    String code = tokenString.substring(prefix.length(), tokenString.length() - suffix.length());
                    int cp = Integer.parseInt(code, 16);
                    tokenString = Character.toString(cp);
                }
            } else {
                tokenString = tokenString.replace('\u2581', ' ');
            }
            sb.append(tokenString);
        }
        return sb.toString();
    }

    public static String replaceControlCharacters(int[] codePoints) {
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4));
            } else {
                chars.appendCodePoint(cp);
            }
        }
        return chars.toString();
    }

    public static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }

}
