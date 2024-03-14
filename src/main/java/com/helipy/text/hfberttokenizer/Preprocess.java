package com.helipy.text.hfberttokenizer;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * @author wangchuangfeng
 */
public class Preprocess {

    public Preprocess() {
    }

    public BiMap<String, Integer> load(String filePath) {
        // 读取词表数据
        try (FileInputStream fileInputStream = new FileInputStream(filePath)) {
            try (InputStreamReader inputStreamReader =
                         new InputStreamReader(fileInputStream, StandardCharsets.UTF_8)) {
                return load(inputStreamReader);
            }
        } catch (IOException e) {
            throw new RuntimeException("read bert vocab error!", e);
        }
    }

    public BiMap<String, Integer> load(InputStreamReader inputStreamReader) {
        BiMap<String, Integer> map = HashBiMap.create();
        // 读取词表数据
        try (BufferedReader br = new BufferedReader(inputStreamReader)) {
            int index = 0;
            String token;
            while ((token = br.readLine()) != null) {
                map.put(token, index);
                index += 1;
            }
            return map;
        } catch (IOException e) {
            throw new RuntimeException("read bert vocab error!", e);
        }
    }

    /**
     * 全角转半角
     */
    public String full2HalfChange(String qjChineseStr) {
        StringBuilder outStrBuf = new StringBuilder();
        String tmpStr = "";
        byte[] b = null;
        try {
            for (int i = 0; i < qjChineseStr.length(); i++) {
                tmpStr = qjChineseStr.substring(i, i + 1);
                if (tmpStr.equals("　")) {
                    outStrBuf.append(" ");
                    continue;
                }
                // "unicode"
                b = tmpStr.getBytes(StandardCharsets.UTF_16);
                // 得到 unicode 字节数据
                if (b[2] == -1) {
                    // 表示全角？
                    b[3] = (byte) (b[3] + 32);
                    b[2] = 0;
                    outStrBuf.append(new String(b, StandardCharsets.UTF_16));
                } else {
                    outStrBuf.append(tmpStr);
                }
            }
        } catch (Exception e) {
            // log.error("全角转半角字符 error. str: {}. error=", qjChineseStr, e);
            return qjChineseStr;
        }
        return outStrBuf.toString();
    }
}
