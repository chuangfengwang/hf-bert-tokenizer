package com.helipy.commons.util.thrift;

import com.alibaba.fastjson2.JSON;
import com.helipy.commons.util.RunUtils;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Pack:       com.helipy.commons.util.thrift
 * File:       KwPodIpPool
 * Desc:
 *
 * @author wangchuangfeng
 * CreateTime: 2024-03-27 17:12
 */
@Slf4j
public abstract class KwPodIpPool {
    private List<String> podIpList;
    private int kwPodIpIdx;
    /**
     * 刷新分钟周期, 在分钟数是 minutePeriod 的整数倍时刷新
     */
    private int minutePeriod;

    public KwPodIpPool(int minutePeriod) {
        this.minutePeriod = minutePeriod;

        // 获取一次
        refreshIpList();
        // 周期刷新
        startSchedule();
        // 初始化 id
        kwPodIpIdx = ThreadLocalRandom.current().nextInt(podIpList.size());
    }

    private void startSchedule() {
        RunUtils.scheduleMinutePeriod(this::refreshIpList, minutePeriod, 2);
    }

    public boolean refreshIpList() {
        // 查所有 pod 的 ip
        List<String> podIpListTmp = getPodIps();
        if (podIpListTmp.isEmpty()) {
            log.warn("KwPodIpPool.refreshIpList error, podIpList is empty, will keep last result");
            return false;
        }
        log.info("KwPodIpPool.refreshIpList success!. podIpList.size:{}, podIpList:{}",
                podIpListTmp.size(), JSON.toJSONString(podIpListTmp));
        this.podIpList = podIpListTmp;
        return true;
    }

    public List<String> getPodIpList() {
        return podIpList;
    }

    public String getNextIp() {
        kwPodIpIdx = (kwPodIpIdx + 1) % podIpList.size();
        return podIpList.get(kwPodIpIdx);
    }

    public abstract List<String> getPodIps();
}
