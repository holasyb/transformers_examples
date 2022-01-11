#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征

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
	BertTokenizer
	)


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
    model = AlbertModel.from_pretrained(model_path)
    return tokenizer, model
	
# tokenizer, model = auto_model()
tokenizer, model = albert_model()


# 编码测试
input_ids = tokenizer.encode(u'语言模型')#, add_special_tokens=True)
input_ids = torch.tensor([input_ids])
print('input_ids', input_ids)

print('\n ===== predicting =====\n')
results = model(input_ids=input_ids)
print(len(results))
# 0: last_hidden_state
# 1: pooler_output
print(results[0].shape)
print(results[1].shape)