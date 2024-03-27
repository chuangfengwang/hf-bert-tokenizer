#!/usr/bin python
# -*- encoding: utf-8 -*-

"""
# File       : log_conf.py
# Description: 
# Author     : wangchuangfeng
# CreateTime : 2024-01-26 15:40
"""
import logging
import os
import sys
from logging import handlers

# 日志目录
log_dir = os.getenv("app_logs", "./logs/")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

info_log_name = "info.log"
error_log_name = "error.log"

# 日志格式
log_fmt = "%(asctime)s.%(msecs)03d [%(process)d-%(thread)d] %(filename)s:%(lineno)d %(funcName)s " \
          "[%(levelname)s] %(message)s"
date_fmt = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(fmt=log_fmt, datefmt=date_fmt)


# filter

# 只保留非错误级别的日志
class NormalLogFilter(logging.Filter):
    def __init__(self, name):
        super().__init__(name=name)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.ERROR:
            return True
        return False


# handler

# 打印到控制台(默认输出到标准错误流 sys.stderr)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)

# info 日志打印到文件
info_log_file_handler = handlers.TimedRotatingFileHandler(
    filename=os.path.join(log_dir, info_log_name), when="midnight", backupCount=15, encoding="utf-8")
info_log_file_handler.setFormatter(formatter)
info_log_file_handler.setLevel(logging.INFO)
normal_filter = NormalLogFilter("normalFilter")
info_log_file_handler.addFilter(normal_filter)

# error 日志打印到文件
error_log_file_handler = handlers.TimedRotatingFileHandler(
    filename=os.path.join(log_dir, error_log_name), when="midnight", backupCount=15, encoding="utf-8")
error_log_file_handler.setFormatter(formatter)
error_log_file_handler.setLevel(logging.ERROR)

# 生成 logger
logger = logging.getLogger("root")
logger.addHandler(console_handler)
logger.addHandler(info_log_file_handler)
logger.addHandler(error_log_file_handler)
logger.setLevel(logging.INFO)


# ========= test code ======== #

def _log_use_test():
    logger.setLevel(logging.DEBUG)
    logger.debug("dddebug")
    logger.info("iiinfo")
    logger.warning("wwwarning")
    logger.error("eeerror")
    logger.critical("cccritical")
    logger.info("占位符用法: %s:%d, 哈哈哈:%f", "词语", 12, 3.01)
    word, idx, weight = "中文", 23, 5.23
    logger.info(f"占位符用法: {word}:{idx}, 哈哈哈:{weight}")

    try:
        12 / 0
    except Exception as e:
        # 使用 logging 参数输出异常
        logger.error("Houston, we have a %s", "major problem", exc_info=True)

    try:
        12 / 0
    except Exception as e:
        # 直接使用 logging 的 exception 函数输出异常堆栈信息
        logger.exception("Houston, we have another %s. error= {}".format(e), "major problem", )

    try:
        12 / 0
    except Exception as e:
        import traceback
        s = traceback.format_exc()
        # 使用 logging + traceback 模块输出异常
        logging.info("traceback: {}".format(s))


if __name__ == "__main__":
    _log_use_test()
