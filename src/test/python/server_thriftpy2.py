import os
import time
from typing import Dict, List, Union

import thriftpy2
from thriftpy2.rpc import make_server
from transformers import AutoTokenizer

from log_conf import logger

interface_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/bert.thrift')
thrift_server_port = int(os.getenv("PORT", 8080))
# module_name 必须以 _thrift 结尾
interface_thrift = thriftpy2.load(interface_path, module_name="hf_bert_tokenizer_thrift")

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

    def bert_tokenizer(self, param: interface_thrift.TokenizerParam,
                       truncate_len: int = 0) -> interface_thrift.TokenizerResult:
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
        result = interface_thrift.TokenizerResult(tokenizer_word_list=tokenizer_word_list,
                                                  tokenizer_id_list=tokenizer_id_list,
                                                  base=interface_thrift.BaseResponse(in_time=in_time))
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


if __name__ == '__main__':
    server = make_server(interface_thrift.BertTokenizer, BertTokenizerHandler(), '0.0.0.0', thrift_server_port)
    print(f'server start at {thrift_server_port}')
    server.serve()
