namespace py hf_bert_tokenizer
namespace java com.helipy.biz

struct BaseResponse {
    1: required i32 status = 0
    2: optional string msg = ""
    3: optional double in_time = 0.0
}

struct TokenizerParam {
    1: required list<string> text_list
    2: optional bool wrapper_id = false
}

struct TokenizerResult {
    1: required list<list<string>> tokenizer_word_list
    2: optional list<list<i32>> tokenizer_id_list
    255: optional BaseResponse base
}

service BertTokenizer {
    string ping(1: string msg)
    TokenizerResult bert_tokenizer(1: TokenizerParam param, 2: i32 truncate_len = 0)
}
