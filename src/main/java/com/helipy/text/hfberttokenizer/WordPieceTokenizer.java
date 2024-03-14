package com.helipy.text.hfberttokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * @author wangchuangfeng
 */
public class WordPieceTokenizer {

    private Map<String, Integer> vocab;

    private String unkToken;
    private int maxInputCharsPerWord;

    public WordPieceTokenizer(Map<String, Integer> vocab, int maxWordLen, String unkToken) {
        this.vocab = vocab;
        this.unkToken = unkToken;
        this.maxInputCharsPerWord = maxWordLen;
    }

    /**
     * For example:
     * input = "unaffable"
     * output = ["un", "##aff", "##able"]
     */
    public List<String> tokenize(String text) {

        List<String> tokens = whiteSpaceTokenize(text);

        List<String> outputTokens = new ArrayList<String>();
        for (String token : tokens) {
            int length = token.length();
            if (length > this.maxInputCharsPerWord) {
                outputTokens.add(this.unkToken);
                continue;
            }

            boolean isBad = false;
            int start = 0;
            List<String> subTokens = new ArrayList<>();

            while (start < length) {
                int end = length;
                String curSubStr = null;
                while (start < end) {
                    String subStr = token.substring(start, end);
                    if (start > 0) {
                        subStr = "##" + subStr;
                    }
                    if (vocab.containsKey(subStr)) {
                        curSubStr = subStr;
                        break;
                    }
                    end -= 1;
                }
                if (null == curSubStr) {
                    isBad = true;
                    break;
                }
                subTokens.add(curSubStr);
                start = end;
            }

            if (isBad) {
                outputTokens.add(unkToken);
            } else {
                outputTokens.addAll(subTokens);
            }

        }
        return outputTokens;
    }

    private List<String> whiteSpaceTokenize(String text) {
        List<String> result = new ArrayList<>();
        if (null == text) {
            return result;
        }
        text = text.trim();
        String[] tokens = text.split("\\s");
        result = Arrays.asList(tokens);

        return result;
    }

}