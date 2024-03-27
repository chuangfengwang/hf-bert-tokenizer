package com.helipy.commons.util.thrift;

import com.alibaba.fastjson2.JSON;
import com.google.common.collect.Lists;
import com.helipy.biz.bert.BertTokenizer;
import com.helipy.biz.bert.TokenizerParam;
import com.helipy.biz.bert.TokenizerResult;
import org.apache.thrift.TException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Pack:       com.helipy.commons.util.thrift
 * File:       ThriftAgentTest
 * Desc:
 *
 * @author wangchuangfeng
 * CreateTime: 2024-03-27 17:51
 */
class ThriftAgentTest {
    private ThriftAgent<BertTokenizer.Client> textThriftAgent;

    private static ExecutorService threadPool = Executors.newFixedThreadPool(3);

    /**
     * 需要启动 thrift server 端测试: 原生 server 方式启动
     */

    @BeforeEach
    void init() {
        // thrift agent 初始化
        textThriftAgent = new ThriftAgent<BertTokenizer.Client>(
                BertTokenizer.Client.class, 8080, 2000, 100_000) {
            @Override
            public List<String> getPodIps() {
                return Lists.newArrayList("127.0.0.1");
            }
        };
        textThriftAgent.init();
    }

    @AfterEach
    void close() {
        textThriftAgent.close();
    }

    @Test
    void pingTest() throws TException {
        String pong = textThriftAgent.getThreadLocalClient().get().getClient().ping("ping消息");
        System.out.println(pong);
    }

    @Test
    void tokenizerTest1() throws TException {
        List<String> textList = Lists.newArrayList(
                "环保工程师,公司主营废气治理、除尘类业务，本人在技术部带领一个小组团队，主要工作内容为： 1.售前技术支持：前期对接客户沟通技术需求点、技术交流、勘察现场收集有效数据； 2.方案设计：根据客户需求及现场情况，组织技术讨论会，评估风险，确认技术点，设计可行的技术方案、图纸，并做成本概算； 3.技术交底：项目进场前对工程部和客户端等进行技术交底； 4.项目验收：项目验收资料编制、汇总，给客户做培训； 5.整个项目生命周期内技术工作跟进、总结； 6.每周对组内成员工作进行汇总并制定工作计划，每月对组内成员进行考核。"
                , "环保工程师,公司主营废气治理、除尘类业务，本人在技术部带领一个小组团队，主要工作内容为： 1.售前技术支持：前期对接客户沟通技术需求点、技术交流、勘察现场收集有效数据； 2.方案设计：根据客户需求及现场情况，组织技术讨论会，评估风险，确认技术点，设计可行的技术方案、图纸，并做成本概算； 3.技术交底：项目进场前对工程部和客户端等进行技术交底； 4.项目验收：项目验收资料编制、汇总，给客户做培训； 5.整个项目生命周期内技术工作跟进、总结； 6.每周对组内成员工作进行汇总并制定工作计划，每月对组内成员进行考核。"
        );
        TokenizerParam param = new TokenizerParam(textList);
        param.setWrapper_id(true);
        TokenizerResult tokenizerResult = textThriftAgent.getThreadLocalClient().get().getClient().bert_tokenizer(param, 0);
        System.out.println(JSON.toJSONString(tokenizerResult.getTokenizer_word_list()));
        System.out.println(JSON.toJSONString(tokenizerResult.getTokenizer_id_list()));
    }

    @Test
    void tokenizerTest2() throws InterruptedException {
        List<CompletableFuture<?>> futureList = new ArrayList<>();
        for (int i = 0; i < 100; ++i) {
            CompletableFuture<?> future = CompletableFuture.runAsync(() -> {
                try {
                    tokenizerTest1();
                } catch (TException e) {
                    e.printStackTrace();
                }
            }, threadPool);
            futureList.add(future);
        }
        CompletableFuture.allOf(futureList.toArray(new CompletableFuture[0])).join();
        Thread.sleep(2000);
    }
}