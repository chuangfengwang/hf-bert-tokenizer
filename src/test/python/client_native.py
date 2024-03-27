import time
from typing import List

from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket
from thrift.transport import TTransport

from hf_bert_tokenizer import BertTokenizer
from hf_bert_tokenizer.ttypes import TokenizerParam, TokenizerResult
from log_conf import logger


class ThriftClientBasic:
    """线程不安全"""

    def __init__(self, host, port, timeout=5000):
        self.socket = TSocket.TSocket(host, port)
        self.socket.setTimeout(ms=timeout)
        self.timeout = timeout
        self.transport = TTransport.TBufferedTransport(self.socket)
        self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        self.client = BertTokenizer.Client(self.protocol)

    def is_open(self):
        return self.transport.isOpen()

    def open(self):
        return self.transport.open()

    def close(self):
        if self.transport.isOpen():
            self.transport.flush()
        self.transport.close()

    def __del__(self):
        self.close()

    def ping(self, text: str):
        try:
            start_time = time.time()
            response: str = self.client.ping(text)
            time_used = time.time() - start_time
            logger.info(f"""thrift rpc api:ping, timeUsed:{time_used}""")
            return response
        except Exception as ex:
            logger.error(f"ping error", exc_info=1)
            raise ex
        finally:
            self.transport.flush()

    def bert_tokenizer(self, text_list: List[str]):
        try:
            start_time = time.time()
            param = TokenizerParam(text_list=text_list, wrapper_id=True)
            response: TokenizerResult = self.client.bert_tokenizer(param, truncate_len=0)
            words_list = response.tokenizer_word_list
            ids_list = response.tokenizer_id_list
            time_used = time.time() - start_time
            logger.info(f"""thrift rpc api:bert_tokenizer, timeUsed:{time_used}""")
            return words_list, ids_list
        except Exception as ex:
            logger.error(f"bert_tokenizer error", exc_info=1)
            raise ex
        finally:
            self.transport.flush()


if __name__ == '__main__':
    host = '127.0.0.1'
    port = 8080
    client = ThriftClientBasic(host=host, port=port)
    client.open()
    # 测试联通性
    pong = client.ping('ping')
    print('ping result:', pong)

    # 测试分词接口
    s = "环保工程师,公司主营废气治理、除尘类业务，本人在技术部带领一个小组团队，主要工作内容为： 1.售前技术支持：前期对接客户沟通技术需求点、技术交流、勘察现场收集有效数据； 2.方案设计：根据客户需求及现场情况，组织技术讨论会，评估风险，确认技术点，设计可行的技术方案、图纸，并做成本概算； 3.技术交底：项目进场前对工程部和客户端等进行技术交底； 4.项目验收：项目验收资料编制、汇总，给客户做培训； 5.整个项目生命周期内技术工作跟进、总结； 6.每周对组内成员工作进行汇总并制定工作计划，每月对组内成员进行考核。"
    texts = [s] * 1
    words_list, ids_list = client.bert_tokenizer(texts)
    for words, ids in zip(words_list, ids_list):
        print('words:', words)
        print('ids:', ids)

    client.close()
