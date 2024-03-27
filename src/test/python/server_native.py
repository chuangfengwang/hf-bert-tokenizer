#!/usr/bin python
# -*- encoding: utf-8 -*-

# thrift 官方原生 server

import json
import multiprocessing as mp
import os
import time
from typing import Dict, List, Union

from thrift.protocol import TBinaryProtocol
from thrift.server import TServer, TProcessPoolServer
from thrift.transport import TSocket
from thrift.transport import TTransport
from transformers import AutoTokenizer

from hf_bert_tokenizer import BertTokenizer
from hf_bert_tokenizer.ttypes import TokenizerParam, TokenizerResult, BaseResponse
from log_conf import logger

thrift_server_port = int(os.getenv("PORT", 8080))
os.environ['RUNNING_MODE'] = 'processpool'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

tokenizer_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/bert')
tokenizer_cache_dir = 'bert-cache'


class BertTokenizerHandler:
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer1 = AutoTokenizer.from_pretrained(tokenizer_config_dir, use_fast=True)
        # self.tokenizer2 = BertTokenizer.from_pretrained(
        #     pretrained_model_name_or_path='bert-base-chinese',  # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
        #     cache_dir=tokenizer_cache_dir,  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
        #     force_download=False,
        # )

    def ping(self, msg: str) -> str:
        return msg

    def bert_tokenizer(self, param: TokenizerParam, truncate_len: int = 0) -> TokenizerResult:
        start = time.time()
        param_text_len_list = []
        tokenizer_word_list = []
        tokenizer_id_list = []
        for text in param.text_list:
            out = self.tokenizer_text(text=text, truncate_len=truncate_len)
            tokenizer_word_list.append(out['input_tokens'])
            if param.wrapper_id:
                tokenizer_id_list.append(out['input_ids'])
            param_text_len_list.append(len(text))
        end = time.time()
        in_time = (end - start) * 1000
        logger.info(f"paramSize:{len(param.text_list)}, textLenSum:{sum(param_text_len_list)}, in_time:{in_time}")
        result = TokenizerResult(tokenizer_word_list=tokenizer_word_list, tokenizer_id_list=tokenizer_id_list,
                                 base=BaseResponse(in_time=in_time))
        return result

    def tokenizer_text(self, text: str, truncate_len: int = 0) -> Dict[str, Union[int, List[int], List[str]]]:
        """
        :param self:
        :param text:
        :param truncate_len:
        :return: dict
            input_ids: token id 编码;
            input_tokens: token 文本词列表;
            token_type_ids: 第一个句子和特殊符号的位置是0，第二个句子的位置是1（含第二个句子末尾的 [SEP]）
            special_tokens_mask: 特殊符号的位置是1，其他位置是0
            attention_mask: pad的位置是0，其他位置是1
            length: 返回句子长度
        """
        out = self.tokenizer1.encode_plus(
            text=text,
            text_pair=None,  # 一次编码两个句子，若没有text_pair这个参数，就一次编码一个句子

            # 当句子长度大于max_length时,截断
            truncation=truncate_len > 0,

            # 一律补零到max_length长度
            padding='max_length',
            max_length=truncate_len,
            add_special_tokens=True,

            # 可取值tensorflow,pytorch,numpy,默认值None为返回list
            return_tensors=None,

            # 返回token_type_ids
            return_token_type_ids=True,

            # 返回attention_mask
            return_attention_mask=True,

            # 返回special_tokens_mask 特殊符号标识
            return_special_tokens_mask=True,

            # 返回offset_mapping 标识每个词的起止位置,这个参数只能 BertTokenizerFast 使用
            # return_offsets_mapping=True,

            # 返回length 标识长度
            return_length=True,
        )
        token_words = self.tokenizer1.decode(out['input_ids']).split(' ')
        out['input_tokens'] = token_words
        return out


class JsonCustomEncoder(json.JSONEncoder):
    def default(self, field):
        if isinstance(field, TokenizerResult):
            return field.__dict__
        elif isinstance(field, BaseResponse):
            return field.__dict__
        else:
            return json.JSONEncoder.default(self, field)


def main_test_handler1():
    tokenizer = BertTokenizerHandler()
    s = "环保工程师,公司主营废气治理、除尘类业务，本人在技术部带领一个小组团队，主要工作内容为： 1.售前技术支持：前期对接客户沟通技术需求点、技术交流、勘察现场收集有效数据； 2.方案设计：根据客户需求及现场情况，组织技术讨论会，评估风险，确认技术点，设计可行的技术方案、图纸，并做成本概算； 3.技术交底：项目进场前对工程部和客户端等进行技术交底； 4.项目验收：项目验收资料编制、汇总，给客户做培训； 5.整个项目生命周期内技术工作跟进、总结； 6.每周对组内成员工作进行汇总并制定工作计划，每月对组内成员进行考核。"
    texts = [s] * 1
    _param = TokenizerParam(text_list=texts, wrapper_id=True)
    _result = tokenizer.bert_tokenizer(_param, truncate_len=0)
    print(json.dumps(_result, ensure_ascii=False, cls=JsonCustomEncoder))


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
