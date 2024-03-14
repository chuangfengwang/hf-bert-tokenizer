package com.helipy.text.hfberttokenizer;

import com.google.common.base.Joiner;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * java 重写的 huggingface transformers BasicTokenizer
 * https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/models/bert/tokenization_bert.py#L371
 *
 * @author wangchuangfeng
 */
@SuppressWarnings("PMD")
public class BasicTokenizer {
    private boolean doLowerCase;
    private Set<String> neverSplitTokens;
    private boolean tokenizeChineseChars;
    private Boolean stripAccents;

    /**
     * 标点字符
     */
    private Pattern punctuationPattern = Pattern.compile("\\pP");
    /**
     * 分隔字符
     */
    private Pattern separatorPattern = Pattern.compile("\\pZ");
    /**
     * 控制字符
     */
    private Pattern controlPattern = Pattern.compile("\\pC");
    /**
     * 空白字符
     */
    private Pattern spacePattern = Pattern.compile("\\s");
    /**
     * 音调字符
     */
    private Pattern accentPattern = Pattern.compile("\\p{Mn}");
    /**
     * 中文字符
     */
    private Pattern chineseCharPattern = Pattern.compile("[\u4e00-\u9fa5]|[\u3400-\u4dbf]");

    public BasicTokenizer() {
        this(true, new HashSet<>(), true, null);
    }

    public BasicTokenizer(boolean doLowerCase, Set<String> neverSplit, boolean tokenizeChineseChars, Boolean stripAccents) {
        this.doLowerCase = doLowerCase;
        this.neverSplitTokens = neverSplit;
        this.tokenizeChineseChars = tokenizeChineseChars;
        this.stripAccents = stripAccents;
    }

    public List<String> tokenize(String text, Set<String> neverSplit) {
        Set<String> neverSplitSet;
        if (neverSplit != null && !neverSplit.isEmpty()) {
            neverSplitSet = new HashSet<>(neverSplit);
        } else {
            neverSplitSet = new HashSet<>(neverSplitTokens);
        }
        String cleanText = cleanText(text);

        if (tokenizeChineseChars) {
            cleanText = tokenizeChineseChars(cleanText);
        }
        List<String> origTokens = whiteSpaceTokenize(cleanText);

        List<String> splitTokens = new ArrayList<>();
        for (String token : origTokens) {
            if (!neverSplitSet.contains(token)) {
                if (doLowerCase) {
                    token = token.toLowerCase();
                    if (!Boolean.FALSE.equals(stripAccents)) {
                        token = runStripAccents(token);
                    }
                } else if (Boolean.TRUE.equals(stripAccents)) {
                    token = runStripAccents(token);
                }
            }
            splitTokens.addAll(runSplitOnPunc(token, new HashSet<>(neverSplitSet)));
        }
        return whiteSpaceTokenize(Joiner.on(" ").join(splitTokens));
    }

    private boolean isControl(char c) {
        if (c == '\t' || c == '\n' || c == '\r') {
            return false;
        }
        return controlPattern.matcher(String.valueOf(c)).matches();
    }

    private boolean isPunctuation(char c) {
        if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64)
                || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) {
            return true;
        }
        return punctuationPattern.matcher(String.valueOf(c)).matches();
    }

    private boolean isWhiteSpace(char c) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            return true;
        }
        return separatorPattern.matcher(String.valueOf(c)).matches();
    }

    private boolean isChineseChar(char c) {
        /* 与 transformers 有一部分 差异
                if ((cp >= 0x4E00 && cp <= 0x9FFF) ||
                        (cp >= 0x3400 && cp <= 0x4DBF) ||
                        (cp >= 0x20000 && cp <= 0x2A6DF) ||
                        (cp >= 0x2A700 && cp <= 0x2B73F) ||
                        (cp >= 0x2B740 && cp <= 0x2B81F) ||
                        (cp >= 0x2B820 && cp <= 0x2CEAF) ||
                        (cp >= 0xF900 && cp <= 0xFAFF) ||
                        (cp >= 0x2F800 && cp <= 0x2FA1F)){
                    return true;
                }
                return false;
        */
        return chineseCharPattern.matcher(String.valueOf(c)).matches();
    }

    private String tokenizeChineseChars(String cleanText) {
        StringBuilder outStrBuf = new StringBuilder();
        for (int i = 0; i < cleanText.length(); i++) {
            char c = cleanText.charAt(i);
            if (isChineseChar(c)) {
                outStrBuf.append(" ");
                outStrBuf.append(c);
                outStrBuf.append(" ");
            } else {
                outStrBuf.append(c);
            }
        }
        return outStrBuf.toString();
    }

    private String cleanText(String text) {
        StringBuilder outStrBuf = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.isHighSurrogate(c) || Character.isLowSurrogate(c)) {
                // utf-16 2 字节无法表示的字符,不参与后续的判断 (一般占 2 个 char)
                outStrBuf.append(c);
                continue;
            }
            if (c == 0 || c == 0xfffd || isControl(c)) {
                continue;
            }
            if (isWhiteSpace(c)) {
                outStrBuf.append(" ");
            } else {
                outStrBuf.append(c);
            }
        }
        return outStrBuf.toString();
    }

    private List<String> whiteSpaceTokenize(String text) {
        List<String> result = new ArrayList<>();
        if (null == text) {
            return result;
        }
        text = text.trim();
        String[] tokens = spacePattern.split(text);
        result = Arrays.stream(tokens).filter(str -> !str.isEmpty()).collect(Collectors.toList());

        return result;
    }


    private List<String> runSplitOnPunc(String token, Set<String> neverSplit) {
        if (neverSplit != null && neverSplit.contains(token)) {
            return Collections.singletonList(token);
        }

        List<List<Character>> charHoldList = new ArrayList<>();
        int length = token.length();
        int i = 0;
        boolean startNewWord = true;
        while (i < length) {
            char c = token.charAt(i);
            if (isPunctuation(c)) {
                List<Character> list = Arrays.asList(c);
                charHoldList.add(list);
                startNewWord = true;
            } else {
                if (startNewWord) {
                    charHoldList.add(new ArrayList<>());
                }
                startNewWord = false;
                charHoldList.get(charHoldList.size() - 1).add(c);
            }
            i += 1;
        }

        List<String> res = new ArrayList<>();
        for (List<Character> characters : charHoldList) {
            StringBuilder sb = new StringBuilder();
            for (Character character : characters) {
                sb.append(character);
            }
            res.add(sb.toString());
        }
        return res;
    }

    private String runStripAccents(String text) {
        String normalizedText = Normalizer.normalize(text, Normalizer.Form.NFD);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < normalizedText.length(); ++i) {
            char c = normalizedText.charAt(i);
            if (accentPattern.matcher(String.valueOf(c)).matches()) {
                continue;
            }
            sb.append(c);
        }
        return sb.toString();
    }

}
