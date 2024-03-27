package com.helipy.commons.util;

import com.alibaba.fastjson2.JSONObject;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.concurrent.BasicThreadFactory;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.FutureTask;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author wangchuangfeng
 * @date 2020-12-14 16:00
 * @description 函数运行机制
 */
@Slf4j
public class RunUtils {
    /**
     * 工具类不需要实例化
     */
    private RunUtils() {
    }

    /**
     * 线程池缓存，避免线程池反复重建
     */
    public static ConcurrentMap<String, ThreadPoolExecutor> poolCache = new ConcurrentHashMap<>();

    public static ThreadPoolExecutor deleteThreadPool(String name) {
        if (!poolCache.containsKey(name)) {
            return null;
        }
        return poolCache.remove(name);
    }

    public static ThreadPoolExecutor createThreadPool(String name, int threadNum) {
        return createThreadPool(name, threadNum, Integer.MAX_VALUE, 60, new LinkedBlockingQueue<>());
    }

    public static ThreadPoolExecutor createThreadPool(String name, int threadNum, int maximumPoolSize,
                                                      int keepAliveTime, BlockingQueue<Runnable> workQueue) {
        if (poolCache.containsKey(name)) {
            ThreadPoolExecutor tp = poolCache.get(name);
            if (!tp.isShutdown() || !tp.isTerminated()) {
                return tp;
            }
        }
        ThreadPoolExecutor tp = new ThreadPoolExecutor(threadNum, maximumPoolSize, keepAliveTime, TimeUnit.SECONDS,
                workQueue, new ThreadFactory() {
            private final AtomicLong idx = new AtomicLong(0);

            @Override
            public Thread newThread(Runnable r) {
                Thread t = new Thread(r, String.format("%s-%02d", name, idx.incrementAndGet()));
                t.setDaemon(true);
                return t;
            }
        });
        tp.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
        tp.allowCoreThreadTimeOut(true);
        poolCache.put(name, tp);
        return tp;
    }

    /**
     * @param func             要执行的逻辑
     * @param retries          失败或超时之后, 还会尝试几次. 也就是, 总共最多会执行 retries + 1 次
     * @param timeoutPerWaitMs 每次运行等待的时间
     * @param <T>              函数返回的对象
     * @return
     * @throws TimeoutException 如果重试次数用完还不能成功,则抛出 TimeoutException 异常
     */
    public static <T> T runWithRetry(Callable<T> func, int retries, long timeoutPerWaitMs) throws TimeoutException {
        return runWithRetry(func, retries, timeoutPerWaitMs, 10L);
    }

    /**
     * @param func             要执行的逻辑
     * @param retries          失败或超时之后, 还会尝试几次. 也就是, 总共最多会执行 retries + 1 次
     * @param timeoutPerWaitMs 每次运行等待的时间
     * @param intervalWait     两次运行之间的等待间隔
     * @param <T>              函数返回的对象
     * @return
     * @throws TimeoutException
     */
    public static <T> T runWithRetry(Callable<T> func, int retries, long timeoutPerWaitMs, long intervalWait) throws TimeoutException {
        FutureTask<T> t = null;
        long start = 0;
        while (true) {
            try {
                t = new FutureTask<>(func);
                t.run();
                /* 如果正常执行则停止 */
                if (timeoutPerWaitMs > 0) {
                    return t.get(timeoutPerWaitMs, TimeUnit.MILLISECONDS);
                } else {
                    return t.get();
                }
            } catch (Exception e) {
                waitAfterExec(start, intervalWait);
                if (t != null) {
                    t.cancel(true);
                }
                retries--;
                if (retries < 0) {
                    TimeoutException exception = new TimeoutException("failed after retry, " + e.getMessage());
                    exception.setStackTrace(e.getStackTrace());
                    log.error("runWithRetry error! error=", e);
                    throw exception;
                }
            }
        }
    }

    public static <T> T runWithRetry(Callable<T> func, int retries) throws TimeoutException {
        return runWithRetry(func, retries, -1);
    }

    public static <T> T runWithRetry(Callable<T> func, int retries, JSONObject cost) throws TimeoutException {
        long now = System.currentTimeMillis();
        T t = runWithRetry(func, retries, -1);
        cost.put("runWithRetry getFeatures", System.currentTimeMillis() - now);
        return t;
    }

    public static Long waitAfterExec(Long time, Long waitDuration) {
        Long cur = System.currentTimeMillis();
        if ((cur - time) < waitDuration) {
            sleep(waitDuration - cur + time);
        }
        return time + waitDuration;
    }

    public static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            log.warn("Thread.sleep() was interrupt. threadName={}", Thread.currentThread().getName());
            Thread.currentThread().interrupt();
        }
    }

    /**
     * 关闭并等待线程池退出
     *
     * @param threadPool   要关闭的线程池
     * @param waitTime     等待时间的数值
     * @param waitTimeUnit 等待时间的单位
     */
    public static void closeAndWaitThreadPool(ThreadPoolExecutor threadPool, long waitTime, TimeUnit waitTimeUnit) {
        threadPool.shutdown();
        try {
            if (!threadPool.awaitTermination(waitTime, waitTimeUnit)) {
                threadPool.shutdownNow();
            }
        } catch (InterruptedException ex) {
            threadPool.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    public static String getStackTrace(Throwable e) {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        e.printStackTrace(pw);
        String sStackTrace = sw.toString();
        return sStackTrace;
    }

    public static void printPoolCache() {
        for (Map.Entry<String, ThreadPoolExecutor> entry : poolCache.entrySet()) {
            String name = entry.getKey();
            ThreadPoolExecutor threadPoolExecutor = entry.getValue();
            log.warn("{} {}", name, threadPoolExecutor);
        }
    }

    /**
     * 每 minutePeriod 分钟执行一次定时任务, 第一次在下一个分钟数为 minutePeriod 的整数倍时执行
     *
     * @param task
     * @param minutePeriod
     */
    public static void scheduleMinutePeriod(Runnable task, int minutePeriod, int corePoolSize) {
        if (minutePeriod < 0 || minutePeriod > 60) {
            throw new RuntimeException("invalid minutePeriod: " + minutePeriod);
        }
        // 下一个整 minutePeriod(例如:10) 分钟开始
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime startTime = now
                .minusSeconds(now.getSecond())
                .minusNanos(now.getNano())
                .minusMinutes(now.getMinute() % minutePeriod)
                .plusMinutes(minutePeriod);
        // minutePeriod 分钟执行一次
        long period = minutePeriod * 60L * 1000L;
        // Date firstTime = Date.from(startTime.toInstant(ZoneOffset.of("+8")));
        // Timer timer = new Timer();
        // timer.scheduleAtFixedRate(task, firstTime, period);
        ScheduledExecutorService executorService = new ScheduledThreadPoolExecutor(corePoolSize,
                new BasicThreadFactory.Builder().namingPattern("scheduleMinutePeriod-%d").daemon(true).build());
        long initialDelay = Duration.between(LocalDateTime.now(), startTime).toMillis();
        executorService.scheduleAtFixedRate(task, initialDelay, period, TimeUnit.MILLISECONDS);
    }
}
