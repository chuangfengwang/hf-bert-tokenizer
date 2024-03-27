package com.helipy.commons.util.thrift;

import com.helipy.commons.util.RunUtils;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeoutException;

/**
 * Pack:       com.helipy.commons.util.thrift
 * File:       ThriftClient
 * Desc:
 *
 * @author wangchuangfeng
 * CreateTime: 2024-03-27 16:52
 */
@Slf4j
public class ThriftClient<Client extends org.apache.thrift.TServiceClient> {
    /**
     * client 具体类型
     */
    private Class<Client> clientClazz;

    @Getter
    private Client client;
    @Getter
    private TTransport transport;

    public ThriftClient(Class<Client> clientClazz) {
        this.clientClazz = clientClazz;
    }

    public ThriftClient open(String serverAddress, int serverPort, int timeout) {
        try {
            this.transport = new TSocket(serverAddress, serverPort, timeout);

            // 协议要和服务端一致
            TProtocol protocol = new TBinaryProtocol(this.transport);
            // TProtocol protocol = new TCompactProtocol(transport);
            // TProtocol protocol = new TJSONProtocol(transport);

            // 用反射方法构造 client
            Constructor constructor = clientClazz.getConstructor(TProtocol.class);
            client = (Client) constructor.newInstance(protocol);

            transport.open();
            log.info("ThriftClient transport.open() success!");
        } catch (TTransportException e) {
            log.error("ThriftClient error TTransportException. server:{}, port:{}, timeout:{}. error=",
                    serverAddress, serverPort, timeout, e);
            transport.close();
            throw new RuntimeException("ThriftClient error TTransportException", e);
        } catch (InvocationTargetException | NoSuchMethodException | InstantiationException | IllegalAccessException e) {
            log.error("ThriftClient create error. server:{}, port:{}. error=",
                    serverAddress, serverPort, e);
            throw new RuntimeException("ThriftClient create error", e);
        }
        return this;
    }

    public void close() {
        flush();
        if (null != transport) {
            transport.close();
        }
    }

    public boolean isOpen() {
        return transport.isOpen();
    }

    public void open() {
        final int retries = 3;
        try {
            RunUtils.runWithRetry((Callable<Void>) () -> {
                transport.open();
                return null;
            }, retries, 50, 5);
        } catch (TimeoutException ex) {
            log.error("thrift transport.open error after retry:{}. error=", retries, ex);
        }
    }

    public void flush() {
        if (null != transport) {
            try {
                transport.flush();
            } catch (TTransportException e) {
                log.warn("thrift transport.flush error. will reopen. error=", e);
                transport.close();
                open();
            }
        }
    }
}
