#!/usr/bin python
# -*- encoding: utf-8 -*-

# thrift 官方原生 server

import multiprocessing as mp
import os

from thrift.protocol import TBinaryProtocol
from thrift.server import TServer, TProcessPoolServer
from thrift.transport import TSocket
from thrift.transport import TTransport

from hf_bert_tokenizer import BertTokenizer
from log_conf import logger
from server_native import BertTokenizerHandler

thrift_server_port = int(os.getenv("PORT", 8080))
os.environ['RUNNING_MODE'] = 'threadpool'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if __name__ == '__main__':
    start_kwargs = {
    }
    # 实现
    handler = BertTokenizerHandler()
    # 服务接口
    processor = BertTokenizer.Processor(handler)
    transport = TSocket.TServerSocket('0.0.0.0', thrift_server_port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    running_mode = os.getenv("RUNNING_MODE", "processpool").lower()
    assert running_mode in {"threaded", "threadpool", "processpool", "simple"}, \
        "Invalid running mode. Running mode should be one of 'threaded', 'threadpool', 'processpool', 'simple'"

    worker_num = int(os.getenv("WORKER_NUM", max(1, mp.cpu_count() - 1)))

    # 多线程模式，每个新连接创建一个新进程
    if running_mode == "threaded":
        server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    # 线程池多线程模式，线程池一开始就创建好固定数目线程
    elif running_mode == "threadpool":
        server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
        # 线程池多线程模式的设置，设置线程数
        server.setNumThreads(worker_num)
    # 多进程模式，每个进程有单独的内存，机器的core越多越快
    elif running_mode == "processpool":
        mp.set_start_method('spawn')
        server = TProcessPoolServer.TProcessPoolServer(processor, transport, tfactory, pfactory)
        # 多进程模式的设置，设置进程数
        server.setNumWorkers(worker_num)
    # 单线程模式（普通模式），一般用于测试调试
    else:
        server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    # 启动服务
    print(f"Start server in {running_mode} mode with work_num {worker_num} in port {thrift_server_port}")
    logger.info(f"Start server in {running_mode} mode with work_num:{worker_num} in port:{thrift_server_port}")
    server.serve()
