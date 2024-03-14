package com.helipy.text.hfberttokenizer;

import com.google.common.base.Strings;
import com.google.common.collect.BiMap;
import com.google.common.collect.Sets;
import com.helipy.text.ahocorasick.DatAutomaton;
import com.helipy.text.ahocorasick.Emit;
import lombok.Getter;
import lombok.Setter;

import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * version1 from: https://github.com/huggingface/tflite-android-transformers/blob/master/bert/src/main/java/co/huggingface/android_transformers/bertqa/tokenization/FullTokenizer.java
 * A java realization of Bert tokenization. Original python code:
 * https://github.com/google-research/bert/blob/master/tokenization.py runs full tokenization to
 * tokenize a String into split subtokens or ids.
 * <p>
 * version2 use djl: https://docs.djl.ai/extensions/tokenizers/index.html
 * HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("./tokenizer.json"))
 */

/**
 * 中文 bert tokenizer
 * 修改自: https://github.com/zhongbin1/bert_tokenization_for_java/blob/master/FullTokenizer.java
 *
 * @author wangchuangfeng
 */
@SuppressWarnings("PMD")
public class FullTokenizer {
    private BiMap<String, Integer> vocab;
    /**
     * 特殊标记符: UNK, CLS, SEP, MASK, PAD
     */
    private Set<String> noSplitTokens;
    /**
     * 特殊标记符: 添加了用户定义的标记
     */
    private Set<String> allSpecialTokens;

    private DatAutomaton<Byte> trie;

    private Preprocess preProcessor;
    private BasicTokenizer basicTokenizer;
    private WordPieceTokenizer wordPieceTokenizer;

    /**
     * 是否执行全角转半角
     */
    private boolean full2Half;
    /**
     * 是否执行转小写
     */
    private boolean doLower;

    @Getter
    @Setter
    private String unkToken = "[UNK]";
    private int unkTokenId;
    @Getter
    @Setter
    private String clsToken = "[CLS]";
    @Getter
    @Setter
    private String sepToken = "[SEP]";
    @Getter
    @Setter
    private String maskToken = "[MASK]";
    @Getter
    @Setter
    private String padToken = "[PAD]";
    private int padTokenId;


    public FullTokenizer(String filePath, boolean full2Half, boolean doLower) {
        this.full2Half = full2Half;
        this.doLower = doLower;
        this.preProcessor = new Preprocess();
        this.vocab = preProcessor.load(filePath);
        this.noSplitTokens = Sets.newHashSet(unkToken, clsToken, sepToken, maskToken, padToken);

        basicTokenizer = new BasicTokenizer();
        wordPieceTokenizer = new WordPieceTokenizer(vocab, 200, unkToken);

        allSpecialTokens = new HashSet<>();
        allSpecialTokens.add(unkToken);
        allSpecialTokens.addAll(noSplitTokens);

        this.unkTokenId = encodeToken(unkToken);
        this.padTokenId = encodeToken(padToken);
        buildTrie();
    }

    private void buildTrie() {
        DatAutomaton.Builder<Byte> builder = DatAutomaton.builder();
        for (String token : allSpecialTokens) {
            builder.add(token);
        }
        trie = builder.build();
    }

    public FullTokenizer(InputStreamReader vocabInputStream, boolean full2Half, boolean doLower) {
        this.full2Half = full2Half;
        this.doLower = doLower;
        this.preProcessor = new Preprocess();
        this.vocab = preProcessor.load(vocabInputStream);
        this.noSplitTokens = Sets.newHashSet(unkToken, clsToken, sepToken, maskToken, padToken);

        basicTokenizer = new BasicTokenizer();
        wordPieceTokenizer = new WordPieceTokenizer(vocab, 200, unkToken);

        allSpecialTokens = new HashSet<>();
        allSpecialTokens.add(unkToken);
        allSpecialTokens.addAll(noSplitTokens);

        this.unkTokenId = encodeToken(unkToken);
        buildTrie();
    }

    List<String> trieSplit(String text) {
        List<String> tokenList = new ArrayList<>();
        if (text == null) {
            return tokenList;
        }
        List<Emit<Byte>> list = trie.parseText(text);
        int idx = 0;
        for (Emit<Byte> emit : list) {
            if (idx < emit.getStart()) {
                String token = text.substring(idx, emit.getStart());
                tokenList.add(token);
            }
            String keyword = text.substring(emit.getStart(), emit.getEnd());
            tokenList.add(keyword);
            idx = emit.getEnd();
        }
        if (idx < text.length()) {
            String token = text.substring(idx);
            tokenList.add(token);
        }
        return tokenList;
    }

    /**
     * 获取分词结果:
     * 中文按字分词,
     * 英文按标点/空白分词, 英文单词再做 wordPiece 分词, wordPiece 分词结果中会包含由 ## 占位的 mask
     *
     * @param text
     * @return
     */
    public List<String> tokenize(String text) {
        List<String> splitTokens = new ArrayList<>();
        // 过滤出 specialTokens
        List<String> tokensSplitBySpecialToken = trieSplit(text);
        for (String piece : tokensSplitBySpecialToken) {
            if (allSpecialTokens.contains(piece)) {
                splitTokens.add(piece);
                continue;
            }

            if (full2Half) {
                piece = preProcessor.full2HalfChange(piece);
            }
            if (doLower) {
                piece = piece.toLowerCase();
            }
            List<String> basicTokenList = basicTokenizer.tokenize(piece, noSplitTokens);
            for (String token : basicTokenList) {
                if (Strings.isNullOrEmpty(token)) {
                    continue;
                }
                if (noSplitTokens.contains(token)) {
                    splitTokens.add(token);
                } else {
                    splitTokens.addAll(wordPieceTokenizer.tokenize(token));
                }
            }
        }
        return splitTokens;
    }

    /**
     * token 列表转 id 列表
     *
     * @param tokens
     * @return
     */
    public List<Integer> convertTokensToIds(List<String> tokens) {
        List<Integer> outputIds = new ArrayList<>(tokens.size());
        for (String token : tokens) {
            outputIds.add(encodeToken(token));
        }
        return outputIds;
    }

    public List<Integer> convertTokensToIds(List<String> tokenList, int maxSeqLength) {
        List<Integer> idList = convertTokensToIds(tokenList);
        List<Integer> idResult = new ArrayList<>(maxSeqLength);
        if (idList.size() < maxSeqLength) {
            idResult.addAll(idList);
            for (int i = idList.size(); i < maxSeqLength; ++i) {
                idResult.add(padTokenId);
            }
        } else if (idList.size() > maxSeqLength) {
            idResult.addAll(idList.subList(0, maxSeqLength));
        } else {
            idResult.addAll(idList);
        }
        return idResult;
    }

    /**
     * id 列表转 token 列表
     *
     * @param ids
     * @return
     */
    public List<String> convertIdsToTokens(List<Integer> ids) {
        List<String> tokens = new ArrayList<>(ids.size());
        for (Integer id : ids) {
            tokens.add(decodeId(id));
        }
        return tokens;
    }

    public int encodeToken(String token) {
        return vocab.getOrDefault(token, unkTokenId);
    }

    public String decodeId(int id) {
        return vocab.inverse().getOrDefault(id, unkToken);
    }

    ///////////////////////////////////////////////////////////////////////////
    // 会添加 [CLS] [SEP] 这种标记 token. 要控制是否添加标记 token, 请使用底层接口:
    // tokenize() convertTokensToIds() encodeToken() decodeId()

    /**
     * 单句映射id
     */
    public Encoding getTokenIdsSingle(List<String> tokensQuery, int maxSeqLength) {
        // truncation： 由于要添加 CLS 和 SEP 两个标记 token，这里要让出 2 个位置
        if (tokensQuery.size() > maxSeqLength - 2) {
            tokensQuery = new ArrayList<>(tokensQuery.subList(0, maxSeqLength - 1));
        }

        List<String> tokens = new ArrayList<>();
        List<Integer> segmentIds = new ArrayList<>();
        tokens.add(clsToken);
        segmentIds.add(0);
        for (String token : tokensQuery) {
            tokens.add(token);
            segmentIds.add(0);
        }
        tokens.add(sepToken);
        segmentIds.add(0);

        List<Integer> inputIds = convertTokensToIds(tokens);
        List<Integer> inputMask = new ArrayList<>();

        for (int i = 0; i < inputIds.size(); i++) {
            inputMask.add(1);
        }

        while (inputIds.size() < maxSeqLength) {
            inputIds.add(0);
            inputMask.add(0);
            segmentIds.add(0);
        }

        return new Encoding(inputIds, inputMask, segmentIds);
    }

    /**
     * 句对映射id
     */
    public Encoding getTokenIdsPair(List<String> tokensQuery, List<String> tokensDoc, int maxSeqLength) {
        // truncation： 由于要添加 CLS，2个SEP 3个标记 token，这里要让出 3 个位置
        if (tokensQuery.size() + tokensDoc.size() > maxSeqLength - 3) {
            int delta = maxSeqLength - 3 - tokensQuery.size() - tokensDoc.size();
            // 差值为偶数，各裁剪一半，为奇数，doc多裁剪1个
            if (delta % 2 == 0) {
                tokensQuery = new ArrayList<>(tokensQuery.subList(0, tokensQuery.size() - delta / 2));
                tokensDoc = new ArrayList<>(tokensDoc.subList(0, tokensDoc.size() - delta / 2));
            } else {
                tokensQuery = new ArrayList<>(tokensQuery.subList(0, tokensQuery.size() - delta / 2));
                tokensDoc = new ArrayList<>(tokensDoc.subList(0, tokensDoc.size() - delta / 2 - 1));
            }
        }

        List<String> tokens = new ArrayList<>();
        List<Integer> segmentIds = new ArrayList<>();
        tokens.add(clsToken);
        segmentIds.add(0);
        for (String token : tokensQuery) {
            tokens.add(token);
            segmentIds.add(0);
        }
        tokens.add(sepToken);
        segmentIds.add(0);

        for (String token : tokensDoc) {
            tokens.add(token);
            segmentIds.add(1);
        }
        tokens.add(sepToken);
        segmentIds.add(1);

        List<Integer> inputIds = convertTokensToIds(tokens);
        List<Integer> inputMask = new ArrayList<>();

        for (int i = 0; i < inputIds.size(); i++) {
            inputMask.add(1);
        }

        while (inputIds.size() < maxSeqLength) {
            inputIds.add(0);
            inputMask.add(0);
            segmentIds.add(0);
        }
        return new Encoding(inputIds, inputMask, segmentIds);
    }

    public Encoding tokenizeSingle(String query, int maxSeqLength) {
        List<String> tokensQuery = tokenize(query);
        return getTokenIdsSingle(tokensQuery, maxSeqLength);
    }

    public Encoding tokenizePair(String query, String doc, int maxSeqLength) {
        List<String> tokensQuery = tokenize(query);
        List<String> tokensDoc = tokenize(doc);
        return getTokenIdsPair(tokensQuery, tokensDoc, maxSeqLength);
    }

    public List<Encoding> tokenizeMultiPairs(String query, List<String> docs, int maxSeqLength) {
        List<String> tokensQuery = tokenize(query);

        List<Encoding> tokenIds = new ArrayList<>();
        for (String doc : docs) {
            List<String> tokensDoc = tokenize(doc);
            Encoding e = getTokenIdsPair(tokensQuery, tokensDoc, maxSeqLength);
            tokenIds.add(e);
        }
        return tokenIds;
    }
}

