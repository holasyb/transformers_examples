#! -*- coding: utf-8 -*-
# 测试代码可用性: 结合MLM的Gibbs采样

import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import (
	AutoTokenizer,
	AutoModel,
	AlbertModel,
	BertTokenizer,
    AlbertForMaskedLM,
    BertForMaskedLM
	)
from tqdm import tqdm

def auto_model():
    """
    自动加载模型的方式
    """
    model_path = '../../pretrained_model/voidful/albert_chinese_small' 
    # 下载：git clone https://huggingface.co/voidful/albert_chinese_tiny
    # git clone https://huggingface.co/voidful/albert_chinese_small
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return tokenizer, model


def albert_model():
    """
    Albert 加载模型的方式
    """
    model_path = '../../pretrained_model/voidful/albert_chinese_small' # 模型路径 
    # 下载：git clone https://huggingface.co/voidful/albert_chinese_tiny
    # git clone https://huggingface.co/voidful/albert_chinese_small
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = AlbertForMaskedLM.from_pretrained(model_path)
    return tokenizer, model

def bert_model():
    """
    Albert 加载模型的方式
    """
    model_path = '../../pretrained_model/hfl/chinese-bert-wwm-ext' # 模型路径 
    # 下载：git clone https://huggingface.co/voidful/albert_chinese_tiny
    # git clone https://huggingface.co/voidful/albert_chinese_small
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path)
    return tokenizer, model

# tokenizer, model = auto_model()
tokenizer, model = bert_model()

# transformers lm 使用示例
# inputtext = "今天[MASK]情很好"
# print(tokenizer.encode(inputtext))
# maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)
# print('maskpos', maskpos)
# from torch.nn.functional import softmax
# input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids, labels=input_ids)
# loss, prediction_scores = outputs[:2]
# logit_prob = softmax(prediction_scores[0, maskpos],dim=-1).data.tolist()
# predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# print(predicted_token, logit_prob[predicted_index])
# print(asdas)



sentences = []
init_sent = u'科学技术是第一生产力。'  # 给定句子或者None
# init_sent = None
minlen, maxlen = 8, 32
steps = 10000
converged_steps = 1000
vocab_size = len(tokenizer)
mask_index = tokenizer.convert_tokens_to_ids('[MASK]')

if init_sent is None:
    length = np.random.randint(minlen, maxlen + 1)
    tokens = ['[CLS]'] + ['[MASK]'] * length + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
else:
    token_ids = tokenizer.encode(init_sent, return_tensors='pt', add_special_tokens=True)
    print('original tokenids', token_ids)
    segment_ids = [[1 for _ in range(len(token_ids[0]))]]
    length = len(token_ids[0]) - 2

softmax = nn.Softmax(dim=0)
with torch.no_grad():
    for _ in tqdm(range(steps), desc='Sampling'):
        # Gibbs采样流程：随机mask掉一个token，然后通过MLM模型重新采样这个token。
        i = np.random.choice(length) + 1
        token_ids[0][i] = mask_index
        probas = model(torch.LongTensor(token_ids), torch.LongTensor(segment_ids))[0][0, i]# .detach().numpy()
        probas = softmax(probas).detach().numpy()
        token = np.random.choice(vocab_size, p=probas)
        token_ids[0][i] = token
        sentences.append(tokenizer.decode(token_ids[0][1:-1]))


print(u'部分随机采样结果：')
for _ in range(10):
    print(np.random.choice(sentences[converged_steps:]))

