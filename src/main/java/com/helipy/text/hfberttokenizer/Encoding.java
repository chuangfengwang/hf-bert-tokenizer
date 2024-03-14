package com.helipy.text.hfberttokenizer;

import java.util.List;

/**
 * @author wangchuangfeng
 * CreateTime: 2023-02-03 17:23
 */
public class Encoding {
    /**
     * token id 序列
     */
    private List<Integer> inputIds;
    /**
     * mask 序列. 实际有字符的位置为 1, padding 字符为 0
     */
    private List<Integer> inputMask;
    /**
     * 片段标记,用于句子对. 第一个句子(包括CLS标记和第一个句子后面的分隔符):0, 第二个句子(包括第二个句子后面的分隔符):1
     */
    private List<Integer> segmentIds;

    public Encoding(List<Integer> inputIds, List<Integer> inputMask, List<Integer> segmentIds) {
        this.inputIds = inputIds;
        this.inputMask = inputMask;
        this.segmentIds = segmentIds;
    }

    public List<Integer> getInputIds() {
        return inputIds;
    }

    public List<Integer> getInputMask() {
        return inputMask;
    }

    public void setInputMask(List<Integer> inputMask) {
        this.inputMask = inputMask;
    }

    public List<Integer> getSegmentIds() {
        return segmentIds;
    }

    public void setSegmentIds(List<Integer> segmentIds) {
        this.segmentIds = segmentIds;
    }

    public void setInputIds(List<Integer> inputIds) {
        this.inputIds = inputIds;
    }
}
