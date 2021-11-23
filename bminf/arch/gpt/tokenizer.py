# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict
from ...utils import jieba

class WordpieceTokenizer(object):

    def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, word):
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]

        start = 0
        sub_tokens = []
        while start < len(word):
            end = len(word)
            cur_substr = None
            while start < end:
                substr = word[start:end]
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                sub_tokens.append(self.unk_token)
                start += 1
                continue
            sub_tokens.append(cur_substr)
            start = end

        return sub_tokens


def Q2B(uchar):
    if uchar == "…":
        return "..."
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: 
        return uchar
    return chr(inside_code)


def read_vocab(path):
    ret = OrderedDict()
    for line in open(path, "r", encoding="utf-8").readlines():
        word = line.strip()
        if len(word) > 0:
            ret[word] = len(ret)
    return ret

class GPT2Tokenizer(object):

    def __init__(self, vocab_path, max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = read_vocab(vocab_path)
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder)

        self.translator_enc = str.maketrans(" \n", "\u2582\u2583")
        self.translator_dec = str.maketrans("\u2582\u2583", " \n")


    @property
    def vocab_size(self):
        return len(self.encoder)

    def __len__(self):
        return len(self.encoder)

    @property
    def eod_id(self):
        return self.encoder[self.eod_token]

    @property
    def pad_id(self):
        return self.encoder[self.pad_token]
    
    @property
    def unk_id(self):
        return self.encoder[self.unk_token]

    @property
    def eod_token(self):
        return '<eod>'

    @property
    def pad_token(self):
        return '<pad>'
    
    @property
    def unk_token(self):
        return "<unk>"
    
    @property
    def start_of_word(self):
        return "\u2581"
        
    def tokenize(self, text):
        """ Tokenize a string. """
        output_tokens = []
        for x in jieba.cut(text, cut_all=False):
            x = self.start_of_word + x.translate(self.translator_enc)
            output_tokens.extend(self.wordpiece_tokenizer.tokenize(x))
        return output_tokens

    def encode(self, text):
        text = ''.join([Q2B(x) for x in text])
        res = self.convert_tokens_to_ids(self.tokenize(text))
        return res

    def decode(self, ids):
        text = ''.join(self.convert_ids_to_tokens(ids))
        text = text.translate(self.translator_dec)
        return text.replace(self.start_of_word, "")

    def convert_tokens_to_ids(self, tokens):
        return [self.encoder.get(x, self.unk_id) for x in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.decoder[x] for x in ids]
