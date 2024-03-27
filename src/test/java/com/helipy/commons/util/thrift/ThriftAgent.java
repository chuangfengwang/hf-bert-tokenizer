package com.helipy.commons.util.thrift;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.pool2.impl.GenericObjectPool;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.time.Duration;
import java.util.List;

/**
 * Pack:       com.helipy.commons.util.thrift
 * File:       ThriftAgent
 * Desc:
 *
 * @author wangchuangfeng
 * CreateTime: 2024-03-27 17:31
 */
@Slf4j
public abstract class ThriftAgent<Client extends org.apache.thrift.TServiceClient> {
    /**
     * client 具体类型
     */
    private Class<Client> clientClazz;

    private int serverPort;
    /**
     * 连接超时时间
     */
    private int connTimeout;
    /**
     * 有一个连接用多久后关闭
     */
    private long expireTimeMs;

    /**
     * 客户端连接池
     */
    private GenericObjectPool<ThriftClient<Client>> thriftClientPool;
    /**
     * socket 线程不安全,包一下
     */
    @Getter
    private ThreadLocal<ThriftClient<Client>> threadLocalClient;

    public ThriftAgent(Class<Client> clientClazz, int serverPort, int connTimeout, long expireTimeMs) {
        this.clientClazz = clientClazz;
        this.serverPort = serverPort;
        this.connTimeout = connTimeout;
        this.expireTimeMs = expireTimeMs;
    }

    public abstract List<String> getPodIps();

    @PostConstruct
    void init() {
        // 1. 构建一个连接池工厂
        PooledThriftClientFactory clientFactory = new PooledThriftClientFactory(
                clientClazz, serverPort, connTimeout, expireTimeMs) {
            @Override
            public List<String> getPodIps() {
                return ThriftAgent.this.getPodIps();
            }
        };

        // 2. 给池子添加支持的配置信息
        GenericObjectPoolConfig<ThriftClient> config = new GenericObjectPoolConfig<>();
        // 2.1 最大池化对象数量
        config.setMaxTotal(12);
        // 2.2 最大空闲池化对象数量
        config.setMaxIdle(12);
        // 2.3 最小空闲池化对象数量
        config.setMinIdle(12);
        // 2.4 间隔多久检查一次池化对象状态,驱逐空闲对象,检查最小空闲数量小于就创建
        config.setTimeBetweenEvictionRuns(Duration.ofSeconds(5));
        // 2.5 阻塞就报错
        config.setBlockWhenExhausted(true);
        // 2.6 最大等待时长超过5秒就报错,如果不配置一直进行等待
        config.setMaxWait(Duration.ofSeconds(5));
        // 2.7 是否开启jmx监控,默认开启
        config.setJmxEnabled(true);
        // 2.8 一定要符合命名规则,否则无效
        config.setJmxNameBase("org.apache.commons.pool2:type=PooledThriftClientFactory,name=ConnectJmxNameBase");
        // 2.9 借出对象时测试可用性. 稳定后关闭
        config.setTestOnBorrow(true);
        // 2.10 创建对象时测试可用性
        config.setTestOnCreate(true);

        thriftClientPool = new GenericObjectPool<>(clientFactory, config);
        // 先进先出. 默认后进先出
        thriftClientPool.setLifo(false);

        threadLocalClient = new ThreadLocal<ThriftClient<Client>>() {
            @Override
            protected ThriftClient<Client> initialValue() {
                try {
                    return thriftClientPool.borrowObject();
                } catch (Exception e) {
                    log.error("thriftClientPool.borrowObject error.", e);
                    throw new RuntimeException("thriftClientPool.borrowObject error", e);
                }
            }

            @Override
            public void remove() {
                thriftClientPool.returnObject(threadLocalClient.get());
                super.remove();
            }
        };
    }

    @PreDestroy
    public void close() {
        thriftClientPool.close();
    }
}
