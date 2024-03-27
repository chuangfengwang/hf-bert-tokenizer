import os

import thrift_connector.connection_pool as connection_pool
import thriftpy2

from hf_bert_tokenizer import BertTokenizer

interface_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/bert.thrift')
interface_thrift = thriftpy2.load(interface_path, module_name="hf_bert_tokenizer_thrift")


def use_thriftpy2_test():
    pool_thrifty2 = connection_pool.ClientPool(
        interface_thrift.BertTokenizer,
        '127.0.0.1',
        8080,
        connection_class=connection_pool.ThriftPyCyClient
    )

    # 测试联通性
    pong = pool_thrifty2.ping('ping')
    print('ping result:', pong)

    # 测试分词接口
    s = "环保工程师,公司主营废气治理、除尘类业务，本人在技术部带领一个小组团队，主要工作内容为： 1.售前技术支持：前期对接客户沟通技术需求点、技术交流、勘察现场收集有效数据； 2.方案设计：根据客户需求及现场情况，组织技术讨论会，评估风险，确认技术点，设计可行的技术方案、图纸，并做成本概算； 3.技术交底：项目进场前对工程部和客户端等进行技术交底； 4.项目验收：项目验收资料编制、汇总，给客户做培训； 5.整个项目生命周期内技术工作跟进、总结； 6.每周对组内成员工作进行汇总并制定工作计划，每月对组内成员进行考核。"
    texts = [s] * 3

    param = interface_thrift.TokenizerParam(text_list=texts, wrapper_id=True)
    response: interface_thrift.TokenizerResult = pool_thrifty2.bert_tokenizer(param, truncate_len=0)

    words_list = response.tokenizer_word_list
    ids_list = response.tokenizer_id_list
    for words, ids in zip(words_list, ids_list):
        print('words:', words)
        print('ids:', ids)


def use_native_test():
    pool_native = connection_pool.ClientPool(
        BertTokenizer,
        '127.0.0.1',
        8080,
        connection_class=connection_pool.ThriftClient
    )

    # 测试联通性
    pong = pool_native.ping('ping')
    print('ping result:', pong)

    # 测试分词接口
    s = "环保工程师,公司主营废气治理、除尘类业务，本人在技术部带领一个小组团队，主要工作内容为： 1.售前技术支持：前期对接客户沟通技术需求点、技术交流、勘察现场收集有效数据； 2.方案设计：根据客户需求及现场情况，组织技术讨论会，评估风险，确认技术点，设计可行的技术方案、图纸，并做成本概算； 3.技术交底：项目进场前对工程部和客户端等进行技术交底； 4.项目验收：项目验收资料编制、汇总，给客户做培训； 5.整个项目生命周期内技术工作跟进、总结； 6.每周对组内成员工作进行汇总并制定工作计划，每月对组内成员进行考核。"
    texts = [s] * 3

    param = interface_thrift.TokenizerParam(text_list=texts, wrapper_id=True)
    response: interface_thrift.TokenizerResult = pool_native.bert_tokenizer(param, truncate_len=0)

    words_list = response.tokenizer_word_list
    ids_list = response.tokenizer_id_list
    for words, ids in zip(words_list, ids_list):
        print('words:', words)
        print('ids:', ids)


if __name__ == '__main__':
    # use_thriftpy2_test()
    use_native_test()
