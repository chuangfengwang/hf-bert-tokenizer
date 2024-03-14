package com.helipy.text.hfberttokenizer;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Pack:       com.helipy.text.hfberttokenizer
 * File:       FullTokenizerTest
 * Desc:
 *
 * @author wangchuangfeng
 * CreateTime: 2024-03-14 23:50
 */
class FullTokenizerTest {
    private FullTokenizer fullTokenizer;

    @BeforeEach
    void init() throws Exception {
        // 词表文件里要有几个特殊 token: [UNK] [CLS] [SEP] [MASK] [PAD]
        String vocabResourcePath = "bert/vocab.txt";
        boolean full2half = false;
        boolean doLower = true;

        // 从 InputStream 加载词表
        try (InputStream inputStream = this.getClass().getClassLoader().getResourceAsStream(vocabResourcePath)) {
            try (InputStreamReader inputStreamReader = new InputStreamReader(inputStream, StandardCharsets.UTF_8)) {
                fullTokenizer = new FullTokenizer(inputStreamReader, full2half, doLower);
            }
        } catch (IOException e) {
            throw new RuntimeException("read bert vocab error!", e);
        }

        // 也可以从文件加载词表(文件不能在 jar 里, 要先解压出来)
        System.out.println(System.getProperty("user.dir"));
        fullTokenizer = new FullTokenizer("src/test/resources/bert/vocab.txt", full2half, doLower);
    }

    @Test
    void tokenize() {
        String text = "美甲\uD83D\uDC85\uD83C\uDFFB。";
        System.out.println(text);
        List<String> tokenList = fullTokenizer.tokenize(text);
        System.out.println(tokenList);
    }

    @Test
    void convertTokensToIds1() {
        // 一般文本
        String text = "要求：有责任心、爱心\uD83D\uDC97、耐心，有教师资格证或\uD83C\uDE36️工作经验的优先入用\\n";
        System.out.println(text);

        // 分词
        List<String> tokenList = fullTokenizer.tokenize(text);
        System.out.println(tokenList);

        // 分词结果转 id
        List<Integer> idList = fullTokenizer.convertTokensToIds(tokenList);
        System.out.println(idList);
    }

    @Test
    void convertTokensToIds2() {
        // 带标记的文本
        String text = "[CLS]美甲。地址: [SEP] message:StatefulSet milk[PAD][PAD][PAD][PAD]";
        System.out.println(text);

        // 分词
        List<String> tokenList = fullTokenizer.tokenize(text);
        System.out.println(tokenList);

        // 分词结果转 id
        List<Integer> idList = fullTokenizer.convertTokensToIds(tokenList);
        System.out.println(idList);
    }

    @Test
    void convertIdsToTokens() {
        String text = "要求：有责任心、爱心\uD83D\uDC97、耐心，有教师资格证或\uD83C\uDE36️工作经验的优先入用\\n";
        System.out.println(text);

        // 分词
        List<String> tokenList = fullTokenizer.tokenize(text);
        System.out.println(tokenList);

        // 得到三个向量输入
        Encoding encoding = fullTokenizer.getTokenIdsSingle(tokenList, 10);
        System.out.println(encoding.getInputIds());
        System.out.println(encoding.getInputMask());
        System.out.println(encoding.getSegmentIds());
    }

    @Test
    void tokenizePair() {
        String query = "要求：有责任心、爱心\uD83D\uDC97、耐心，有教师资格证或\uD83C\uDE36️工作经验的优先入用\\n";
        String doc = "美甲。地址: message:StatefulSet milk[PAD][PAD][PAD][PAD]";

        // 分词
        List<String> queryTokenList = fullTokenizer.tokenize(query);
        List<String> docTokenList = fullTokenizer.tokenize(doc);
        List<String> tokenList = new ArrayList<>(queryTokenList.size() + docTokenList.size());
        tokenList.addAll(queryTokenList);
        tokenList.addAll(docTokenList);
        System.out.println(tokenList);

        // 得到三个向量输入
        Encoding encoding = fullTokenizer.getTokenIdsPair(queryTokenList, docTokenList, 80);
        System.out.println(encoding.getInputIds());
        System.out.println(encoding.getInputMask());
        System.out.println(encoding.getSegmentIds());
    }
}