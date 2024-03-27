package com.helipy.commons.util.thrift;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.pool2.DestroyMode;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.PooledObjectFactory;
import org.apache.commons.pool2.impl.DefaultPooledObject;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Pack:       com.helipy.commons.util.thrift
 * File:       PooledThriftClientFactory
 * Desc:
 *
 * @author wangchuangfeng
 * CreateTime: 2024-03-27 17:11
 */
@Slf4j
public abstract class PooledThriftClientFactory<Client extends org.apache.thrift.TServiceClient>
        implements PooledObjectFactory<ThriftClient<Client>> {

    /**
     * client 具体类型
     */
    private Class<Client> clientClazz;

    private final int serverPort;
    /**
     * 连接等待响应的时间
     */
    private final int timeoutMs;
    /**
     * 一个连接的过期时间,过期后重连 毫秒
     */
    private final long expireTimeMs;

    private final KwPodIpPool kwPodIpPool;

    public abstract List<String> getPodIps();

    public PooledThriftClientFactory(
            Class<Client> clientClazz, int serverPort, int timeoutMs, long expireTimeMs) {
        this.clientClazz = clientClazz;

        this.serverPort = serverPort;
        this.timeoutMs = timeoutMs;
        this.expireTimeMs = expireTimeMs;

        kwPodIpPool = new KwPodIpPool(10) {
            @Override
            public List<String> getPodIps() {
                return PooledThriftClientFactory.this.getPodIps();
            }
        };
    }


    @Override
    public void activateObject(PooledObject<ThriftClient<Client>> pooledObject) throws Exception {
    }

    /**
     * 销毁连接
     *
     * @param pooledObject
     * @throws Exception
     */
    @Override
    public void destroyObject(PooledObject<ThriftClient<Client>> pooledObject) throws Exception {
        ThriftClient thriftClient = pooledObject.getObject();
        thriftClient.close();
    }

    @Override
    public void destroyObject(PooledObject<ThriftClient<Client>> p, DestroyMode destroyMode) throws Exception {
        PooledObjectFactory.super.destroyObject(p, destroyMode);
    }

    /**
     * 创建一个连接
     *
     * @return
     * @throws Exception
     */
    @Override
    public PooledObject<ThriftClient<Client>> makeObject() throws Exception {
        // 先查所有 pod 的 ip, 轮询选一个
        List<String> podIpList = kwPodIpPool.getPodIpList();
        if (podIpList.isEmpty()) {
            throw new RuntimeException("TextThriftAgent makeObject error, podIpList is empty");
        }
        String serverAddressUsed = kwPodIpPool.getNextIp();
        log.info("TextThriftAgent makeObject, serverAddress:{}", serverAddressUsed);

        ThriftClient<Client> thriftClient = new ThriftClient<>(clientClazz);
        thriftClient.open(serverAddressUsed, serverPort, timeoutMs);
        return new DefaultPooledObject<>(thriftClient);
    }

    /**
     * 回收资源时候进行调用
     *
     * @param pooledObject
     * @throws Exception
     */
    @Override
    public void passivateObject(PooledObject<ThriftClient<Client>> pooledObject) throws Exception {
        Instant createAtInstant = pooledObject.getCreateInstant();
        LocalDateTime createTime = LocalDateTime.ofInstant(createAtInstant, ZoneId.systemDefault());
        Duration duration = Duration.between(createTime, LocalDateTime.now());
        int random = ThreadLocalRandom.current().nextInt(1000);
        if (duration.toMillis() >= expireTimeMs + random) {
            // 过期销毁
            destroyObject(pooledObject);
        }
    }

    /**
     * 是否可用
     *
     * @param pooledObject
     * @return
     */
    @Override
    public boolean validateObject(PooledObject<ThriftClient<Client>> pooledObject) {
        ThriftClient thriftClient = pooledObject.getObject();
        return thriftClient.isOpen();
    }
}
