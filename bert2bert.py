import argparse
import glob
import json
import logging
import os
import pickle
import random
import math
import re
import shutil
import sys
from typing import Dict, List, Tuple
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
)

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def normalized_uniform_init(w):
    init_weight = torch.rand_like(w)
    # nn.init.uniform_(init_weight, 0.0, 1.0)
    # if 'softmax' in init_scheme:
    #     init_weight = F.softmax(init_weight, -1) # softmax normalize
    # else:
    init_weight = init_weight / torch.sum(init_weight, -1, keepdim=True)  # normalize
    w.copy_(init_weight)


def wider3d(w, dim, new_width, choices, div=False, add_noise=False):
    old_width = w.size(dim)
    if dim == 0:
        # new_w = torch.randn(new_width, w.size(1), w.size(2), dtype=torch.float16, device='cuda')
        new_w = torch.randn(new_width, w.size(1), w.size(2), device='cuda')
    elif dim == 1:
        # new_w = torch.randn(w.size(0), new_width, w.size(2), dtype=torch.float16, device='cuda')
        new_w = torch.randn(w.size(0), new_width, w.size(2), device='cuda')
    else:
        # new_w = torch.randn(w.size(0), w.size(1), new_width, dtype=torch.float16, device='cuda')
        new_w = torch.randn(w.size(0), w.size(1), new_width, device='cuda')
    new_w.narrow(dim, 0, old_width).copy_(w.clone())
    tracking = dict()
    for i in range(old_width, new_width):
        idx = choices[i - old_width]
        try:
            tracking[idx].append(i)
        except:
            tracking[idx] = [idx]
            tracking[idx].append(i)

        new_w.select(dim, i).copy_(w.select(dim, idx).clone())
    if div:
        if dim == 0:
            for idx, d in tracking.items():
                flag = False
                for item in d:
                    if flag == False:
                        flag = True
                        new_w[item].div_(len(d))
                    else:
                        new_w[item].div_(len(d))
                        if add_noise:
                            noise = torch.empty(new_w[item].size(), dtype=torch.float16, device='cuda')
                            nn.init.normal_(noise, std=0.01)
                            new_w[item] += noise
        elif dim == 1:
            for idx, d in tracking.items():
                flag = False
                for item in d:
                    if flag == False:
                        flag = True
                        new_w[:, item].div_(len(d))
                    else:
                        new_w[:, item].div_(len(d))
                        if add_noise:
                            noise = torch.empty(new_w[:, item].size(), dtype=torch.float16, device='cuda')
                            nn.init.normal_(noise, std=0.01)
                            new_w[:, item] += noise
        else:
            for idx, d in tracking.items():
                flag = False
                for item in d:
                    if flag == False:
                        flag = True
                        new_w[:, :, item].div_(len(d))
                    else:
                        new_w[:, :, item].div_(len(d))
                        if add_noise:
                            noise = torch.empty(new_w[:, :, item].size(), dtype=torch.float16, device='cuda')
                            nn.init.normal_(noise, std=0.01)
                            new_w[:, :, item] += noise

    # return new_w.half()
    return new_w


def wider2d(w, dim, new_width, choices, div=False, add_noise=False):
    old_width = w.size(dim)
    if dim == 0:
        # new_w = torch.randn(new_width, w.size(1), dtype=torch.float16, device='cuda')
        new_w = torch.randn(new_width, w.size(1), device='cuda')
    else:
        # new_w = torch.randn(w.size(0), new_width, dtype=torch.float16, device='cuda')
        new_w = torch.randn(w.size(0), new_width, device='cuda')
    new_w.narrow(dim, 0, old_width).copy_(w.clone())
    tracking = dict()
    for i in range(old_width, new_width):
        idx = choices[i - old_width]
        try:
            tracking[idx].append(i)
        except:
            tracking[idx] = [idx]
            tracking[idx].append(i)

        new_w.select(dim, i).copy_(w.select(dim, idx).clone())
    if div:
        if dim == 0:
            for idx, d in tracking.items():
                flag = False
                for item in d:
                    if flag == False:
                        flag = True
                        new_w[item].div_(len(d))
                    else:
                        new_w[item].div_(len(d))
                        if add_noise:
                            noise = torch.empty(new_w[item].size(), dtype=torch.float16, device='cuda')
                            nn.init.normal_(noise, std=0.01)
                            new_w[item] += noise
        else:
            for idx, d in tracking.items():
                flag = False
                for item in d:
                    if flag == False:
                        flag = True
                        new_w[:, item].div_(len(d))
                    else:
                        new_w[:, item].div_(len(d))
                        if add_noise:
                            noise = torch.empty(new_w[:, item].size(), dtype=torch.float16, device='cuda')
                            nn.init.normal_(noise, std=0.01)
                            new_w[:, item] += noise
    # return new_w.half()
    return new_w


def wider(w, new_width, choices, div=False, add_noise=False):
    old_width = w.size(0)
    # new_w = torch.randn(new_width, dtype=torch.float16, device='cuda')
    new_w = torch.randn(new_width, device='cuda')
    new_w.narrow(0, 0, old_width).copy_(w.clone())  # 把旧参数copy过去，new_w的shape不变，其余参数仍是random
    tracking = dict()
    for i in range(old_width, new_width):
        idx = choices[i - old_width]
        try:
            tracking[idx].append(i)  # idx是copy位置，i是新参数的序号
        except:
            tracking[idx] = [idx]
            tracking[idx].append(i)

        new_w.select(0, i).copy_(w.select(0, idx).clone())  # 将旧参数中对应idx的参数，copy到新参数右边的位置
    if div:
        for idx, d in tracking.items():
            flag = False
            for item in d:
                if flag == False:
                    flag = True
                    new_w[item].div_(len(d))  # 对新权重进行放缩？只用于ln层？
                else:
                    new_w[item].div_(len(d))
                    if add_noise:
                        # noise = torch.empty(new_w[item].size(), dtype=torch.float16, device='cuda')
                        noise = torch.empty(new_w[item].size(), device='cuda')
                        nn.init.normal_(noise, std=0.01)
                        new_w[item] += noise
    # return new_w.half()  # 半精度？
    return new_w


def get_choices(old_width, new_width, is_always_left=False):
    choices = []
    if is_always_left:
        idx = 0
        for i in range(old_width, new_width):
            choices.append(idx)
            idx += 1
    else:
        for i in range(old_width, new_width):
            idx = np.random.randint(0, old_width)
            choices.append(idx)
    return choices


@torch.no_grad()  # 有必要吗？
def FPI(args, model_large, bm_layers, bm_hidden, bm_num_heads, bm_intermediate_size, add_noise=False, add_last=False, source_model=None):
    # bm对应大模型，sm对应小模型？
    # ckpt = torch.load(path)  # 加载小模型？
    # load source model from the setting
    if source_model is None:
        assert len(args.source_model_path) == 1, 'Not support multiple model.'

        source_model_path = args.source_model_path[0]
        # Small model
        if source_model_path:
            small_config = AutoConfig.from_pretrained(source_model_path, cache_dir=args.cache_dir)
        else:
            raise ValueError("No config for small model is specified.")
        model_small = model_large.__class__.from_pretrained(
            source_model_path,
            from_tf=bool(".ckpt" in source_model_path),
            config=small_config,
            cache_dir=args.cache_dir,
        )
    dict_model_small = model_small.state_dict()
    # random.seed(ckpt['args'].seed)
    # np.random.seed(ckpt['args'].seed)
    sm_layers = 3
    sm_hidden = 256
    sm_num_heads = 4
    sm_intermediate_size = 1024
    # sm_layers = ckpt['args'].encoder_layers
    # sm_hidden = ckpt['args'].encoder_embed_dim
    # sm_num_heads = ckpt['args'].encoder_attention_heads
    # sm_intermediate_size = ckpt['args'].encoder_ffn_embed_dim
    # is_always_left = ckpt['args'].is_always_left
    # layer_candidates = ckpt['args'].layer_candidates
    # layer_idxs = ckpt['args'].layer_idxs  # 这俩有啥用啊？
    layer_candidates = None
    layer_idxs = None
    headdim = bm_hidden // bm_num_heads

    if layer_candidates == None:
        layer_candidates = [i for i in range(sm_layers)]
    if layer_idxs == None:
        layer_idxs = [i for i in range(sm_layers)]  # 这俩变量怎么相等啊？是较小模型的layer index？

    added_layers = []
    added_layer_num = bm_layers - sm_layers
    print("number of added layer: " + str(added_layer_num))
    print("old layer candidates: " + str(layer_candidates))
    print("old layer idxs: " + str(layer_idxs))
    if len(layer_candidates) >= added_layer_num:  # 比如从6层grow到12层
        added_layers = layer_candidates[-added_layer_num:] if add_last else random.sample(layer_candidates,
                                                                                          added_layer_num)
        for layer in added_layers:  # added_layers是
            layer_candidates.remove(layer)  # layer_candidates这不就为空了吗？
    else:
        while added_layer_num > len(layer_candidates):
            added_layers = added_layers + layer_candidates.copy()
            added_layer_num -= len(layer_candidates)
            layer_candidates = [i for i in range(max(layer_idxs) + 1)]
        new_added_layers = layer_candidates[-added_layer_num:] if add_last else random.sample(layer_candidates,
                                                                                              added_layer_num)
        for layer in new_added_layers:
            layer_candidates.remove(layer)
        added_layers = added_layers + new_added_layers

    sm_layer_idxs = [i for i in range(sm_layers)]
    sm_layer_idx_for_bert2bert_top = []
    new_layer_idxs = []
    for layer in sm_layer_idxs:
        idx = layer_idxs[layer]
        sm_layer_idx_for_bert2bert_top.append(layer)
        new_layer_idxs.append(idx)
        while idx in added_layers:
            sm_layer_idx_for_bert2bert_top.append(layer)
            new_layer_idxs.append(idx)
            added_layers.remove(idx)
    assert len(new_layer_idxs) == bm_layers
    assert len(new_layer_idxs) == len(sm_layer_idx_for_bert2bert_top)
    print("new layer candidates: " + str(layer_candidates))
    print("new layer idxs: " + str(new_layer_idxs))
    if len(new_layer_idxs) % (max(layer_idxs) + 1) == 0:
        new_layer_idxs = [i for i in range(bm_layers)]  # 又改成了大模型的实际index
        layer_candidates = [i for i in range(bm_layers)]

    print("final layer candidates: " + str(layer_candidates))
    print("final layer idxs: " + str(new_layer_idxs))
    print(
        "sm layer idx for bert2bert: " + str(sm_layer_idx_for_bert2bert_top))  # # 大模型的layer index，其中index都是小模型复制后的index
    new_layer_idxs = [0, 1, 2, 0, 1, 2]
    sm_layer_idx_for_bert2bert_top = [0, 1, 2, 0, 1, 2]
    choose_hidden_dims = get_choices(sm_hidden, bm_hidden, is_always_left=False)

    lst = []

    # print('*' * 500)
    # for k, v in dict_model_small.items():
    #     print(k, v.shape)
    for k, v in dict_model_small.items():  # 在宽度上沿dim0劈两半？采用不同学习率？或者用一个小模型去对比，
        # if 'embed_tokens' in k or 'embed_position' in k or 'domain_embeddings' in k:
        #     new_weight = wider2d(v, 1, bm_hidden, choose_hidden_dims, add_noise=add_noise)
        # elif 'emb_layer_norm' in k:
        #     new_weight = wider(v, bm_hidden, choose_hidden_dims, add_noise=add_noise)
        if 'word_embeddings' in k or 'position_embeddings' in k or 'token_type_embeddings' in k:
            new_weight = wider2d(v, 1, bm_hidden, choose_hidden_dims, add_noise=add_noise)
        elif 'LayerNorm' in k:
            new_weight = wider(v, bm_hidden, choose_hidden_dims, add_noise=add_noise)
        elif 'cls' in k:
            if 'dense' in k:
                if 'weight' in k:
                    new_weight = wider2d(wider2d(v, 1, bm_hidden, choose_hidden_dims, div=True, add_noise=add_noise), 0,
                                         bm_hidden, choices=choose_hidden_dims, add_noise=add_noise)
                elif 'bias' in k:
                    new_weight = wider(v, bm_hidden, choose_hidden_dims, add_noise=add_noise)
            elif 'LayerNorm' in k:
                new_weight = wider(v, bm_hidden, choose_hidden_dims, div=True, add_noise=add_noise)
            elif 'weight' in k:
                new_weight = wider2d(v, 1, bm_hidden, choose_hidden_dims, div=True, add_noise=add_noise)
            elif 'bias' in k:
                new_weight = v
        elif 'position_ids' in k:
            new_weight = v
        # print(k)
        lst.append([k, new_weight.clone()])  # 前面的key没有一个符合要求的？

    # 到这里宽度就扩展完了？然后是深度扩展？
    for bm_layer_idx in range(bm_layers):  # 对12层的每一层？也就是old layer也需要处理？
        sm_layer_idx = sm_layer_idx_for_bert2bert_top[bm_layer_idx]
        # 除了要把旧层copy给新层，还要对所有层都增加num_heads?
        choose_heads = get_choices(sm_num_heads, bm_num_heads, is_always_left=False)
        choose_mlp_dims = get_choices(sm_intermediate_size, bm_intermediate_size, is_always_left=False)
        # 为什么在这里增长MLP的dim？这不是属于宽度吗？
        # layer = f'decoder.sentence_encoder.layers.{sm_layer_idx}'
        # new_layer = f'decoder.sentence_encoder.layers.{bm_layer_idx}'
        layer = f'{sm_layer_idx}'
        new_layer = f'{bm_layer_idx}'
        # print('&'*200)
        # print(layer)
        # print(new_layer)  # 旧的三层，和新的6层，是112233的继承关系吗？
        # logger.info('&'*200)
        # logger.info(layer)
        # logger.info(new_layer)

        # self attention
        for w in ['query', 'key', 'value']:
            # k = f'{layer}.self_attn.{w}.weight'
            # v = ckpt['model'][k]
            k = f'bert.encoder.layer.{layer}.attention.self.{w}.weight'
            v = dict_model_small[k]  # 这里Roberta的name和bert是不是不同啊？

            # new_weight = torch.zeros((bm_hidden, bm_hidden), dtype=torch.float16, device='cuda')
            new_weight = torch.zeros((bm_hidden, bm_hidden), device='cuda')
            new_weight.reshape(bm_num_heads, headdim, bm_hidden).permute(0, 2, 1).copy_(wider3d(
                wider2d(v, 1, bm_hidden, choices=choose_hidden_dims, div=True, add_noise=add_noise).reshape(
                    sm_num_heads, headdim, bm_hidden).permute(0, 2, 1), 0, bm_num_heads, choices=choose_heads,
                add_noise=add_noise))  # 先扩展为(384, 768), 再从[6, 768, 64]扩展为[12, 768, 64]
            # 这我到时候怎么把参数抠出来啊？先抠出来(6, 768, 64)，再抠出来(384, 384)？
            # print('=='*200)
            # print(new_weight.shape)

            # new_k = f'{new_layer}.self_attn.{w}.weight'
            new_k = f'bert.encoder.layer.{new_layer}.attention.self.{w}.weight'
            # print(new_k)
            lst.append([new_k, new_weight.clone()])

            # k = f'{layer}.self_attn.{w}.bias'
            # v = ckpt['model'][k]
            k = f'bert.encoder.layer.{layer}.attention.self.{w}.bias'
            v = dict_model_small[k]

            # new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
            new_weight = torch.zeros(bm_hidden, device='cuda')
            new_weight.reshape(bm_num_heads, headdim).copy_(
                wider2d(v.reshape(sm_num_heads, headdim), 0, bm_num_heads, choose_heads, add_noise=add_noise))

            # new_k = f'{new_layer}.self_attn.{w}.bias'
            new_k = f'bert.encoder.layer.{new_layer}.attention.self.{w}.bias'
            # print(new_k)
            lst.append([new_k, new_weight.clone()])

        # k = f'{layer}.self_attn.out_proj.weight'  # 这个矩阵为什么要单独处理？维度不同吗？
        # v = ckpt['model'][k]
        k = f'bert.encoder.layer.{layer}.attention.output.dense.weight'
        v = dict_model_small[k]
        # new_weight = torch.zeros((bm_hidden, bm_hidden), dtype=torch.float16, device='cuda')
        new_weight = torch.zeros((bm_hidden, bm_hidden), device='cuda')
        new_weight.reshape(bm_hidden, bm_num_heads, headdim).permute(1, 2, 0).copy_(wider3d(
            wider2d(v, 0, bm_hidden, choose_hidden_dims, add_noise=add_noise).reshape(bm_hidden, sm_num_heads,
                                                                                      headdim).permute(1, 2, 0), 0,
            bm_num_heads, choose_heads, div=True, add_noise=add_noise))  # new_weights的shape仍是二维的

        # new_k = f'{new_layer}.self_attn.out_proj.weight'  # 需要先转换shape，然后再收集新参数？
        new_k = f'bert.encoder.layer.{new_layer}.attention.output.dense.weight'
        # print(new_k)
        lst.append([new_k, new_weight.clone()])

        # k = f'{layer}.self_attn.out_proj.bias'
        # v = ckpt['model'][k]  # 观察v对应的old shape和new_weight对应的new shape
        k = f'bert.encoder.layer.{layer}.attention.output.dense.bias'
        v = dict_model_small[k]
        # new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
        new_weight = torch.zeros(bm_hidden, device='cuda')
        new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims, add_noise=add_noise))

        # new_k = f'{new_layer}.self_attn.out_proj.bias'
        new_k = f'bert.encoder.layer.{new_layer}.attention.output.dense.bias'
        # print(new_k)
        lst.append([new_k, new_weight.clone()])

        # k = f'{layer}.self_attn_layer_norm.weight'
        # v = ckpt['model'][k]
        k = f'bert.encoder.layer.{layer}.attention.output.LayerNorm.weight'
        v = dict_model_small[k]
        # new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
        new_weight = torch.zeros(bm_hidden, device='cuda')
        new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims, add_noise=add_noise))

        # new_k = f'{new_layer}.self_attn_layer_norm.weight'
        new_k = f'bert.encoder.layer.{new_layer}.attention.output.LayerNorm.weight'
        # print(new_k)
        lst.append([new_k, new_weight.clone()])

        # k = f'{layer}.self_attn_layer_norm.bias'
        # v = ckpt['model'][k]
        k = f'bert.encoder.layer.{layer}.attention.output.LayerNorm.bias'
        v = dict_model_small[k]
        # new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
        new_weight = torch.zeros(bm_hidden, device='cuda')
        new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims, add_noise=add_noise))

        # new_k = f'{new_layer}.self_attn_layer_norm.bias'
        new_k = f'bert.encoder.layer.{new_layer}.attention.output.LayerNorm.bias'
        # print(new_k)
        lst.append([new_k, new_weight.clone()])

        # ffn
        # k = f'{layer}.fc1.weight'
        # v = ckpt['model'][k]
        k = f'bert.encoder.layer.{layer}.intermediate.dense.weight'
        v = dict_model_small[k]
        # new_weight = torch.zeros(bm_intermediate_size, bm_hidden, dtype=torch.float16, device='cuda')
        new_weight = torch.zeros(bm_intermediate_size, bm_hidden, device='cuda')
        new_weight.copy_(wider2d(wider2d(v, 1, bm_hidden, choose_hidden_dims, div=True, add_noise=add_noise), 0,
                                 bm_intermediate_size, choose_mlp_dims, add_noise=add_noise))

        # new_k = f'{new_layer}.fc1.weight'
        new_k = f'bert.encoder.layer.{new_layer}.intermediate.dense.weight'
        # print(new_k)
        lst.append([new_k, new_weight.clone()])

        # k = f'{layer}.fc1.bias'
        # v = ckpt['model'][k]
        k = f'bert.encoder.layer.{layer}.intermediate.dense.bias'
        v = dict_model_small[k]
        # new_weight = torch.zeros(bm_intermediate_size, dtype=torch.float16, device='cuda')
        new_weight = torch.zeros(bm_intermediate_size, device='cuda')
        new_weight.copy_(wider(v, bm_intermediate_size, choose_mlp_dims, add_noise=add_noise))

        # new_k = f'{new_layer}.fc1.bias'
        new_k = f'bert.encoder.layer.{new_layer}.intermediate.dense.bias'
        # print(new_k)
        lst.append([new_k, new_weight.clone()])

        # k = f'{layer}.fc2.weight'
        # v = ckpt['model'][k]
        k = f'bert.encoder.layer.{layer}.output.dense.weight'
        v = dict_model_small[k]
        # new_weight = torch.zeros(bm_hidden, bm_intermediate_size, dtype=torch.float16, device='cuda')
        new_weight = torch.zeros(bm_hidden, bm_intermediate_size, device='cuda')
        new_weight.copy_(
            wider2d(wider2d(v, 0, bm_hidden, choose_hidden_dims, add_noise=add_noise), 1, bm_intermediate_size,
                    choose_mlp_dims, div=True, add_noise=add_noise))

        # new_k = f'{new_layer}.fc2.weight'
        new_k = f'bert.encoder.layer.{new_layer}.output.dense.weight'
        # print(new_k)
        lst.append([new_k, new_weight.clone()])

        # k = f'{layer}.fc2.bias'
        # v = ckpt['model'][k]
        k = f'bert.encoder.layer.{layer}.output.dense.bias'
        v = dict_model_small[k]
        # new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
        new_weight = torch.zeros(bm_hidden, device='cuda')
        new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims, add_noise=add_noise))

        # new_k = f'{new_layer}.fc2.bias'
        new_k = f'bert.encoder.layer.{new_layer}.output.dense.bias'
        # print(new_k)
        lst.append([new_k, new_weight.clone()])

        # layer_norm
        # k = f'{layer}.final_layer_norm.weight'
        # v = ckpt['model'][k]
        k = f'bert.encoder.layer.{layer}.output.LayerNorm.weight'
        v = dict_model_small[k]
        # new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
        new_weight = torch.zeros(bm_hidden, device='cuda')
        new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims, add_noise=add_noise))

        # new_k = f'{new_layer}.final_layer_norm.weight'
        new_k = f'bert.encoder.layer.{new_layer}.output.LayerNorm.weight'
        # print(new_k)
        lst.append([new_k, new_weight.clone()])

        # k = f'{layer}.final_layer_norm.bias'
        # v = ckpt['model'][k]
        k = f'bert.encoder.layer.{layer}.output.LayerNorm.bias'
        v = dict_model_small[k]
        # new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
        new_weight = torch.zeros(bm_hidden, device='cuda')
        new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims, add_noise=add_noise))

        # new_k = f'{new_layer}.final_layer_norm.bias'
        new_k = f'bert.encoder.layer.{new_layer}.output.LayerNorm.bias'
        # print(new_k)
        lst.append([new_k, new_weight.clone()])

    dict_model_large = {}
    for k, v in lst:
        # noise = torch.zeros_like(v, dtype=torch.float16, device='cuda')
        noise = torch.zeros_like(v, device='cuda')
        if add_noise:
            nn.init.normal_(noise, std=0.01)

        # ckpt['model'][k] = v + noise  # 说明lst中所有的key，都是新加入的参数
        # print(k, v.device)
        dict_model_large[k] = v.cuda() + noise

    # # 随机初始化new layers
    # for k, v in dict_model_large.items():
    #     # print(k, v.shape)
    #     if k[19] in ['1', '3', '5']:
    #         # print('#')
    #         # print(k, v.shape)
    #         logger.info(k)
    #         logger.info(v.shape)
    #         normalized_uniform_init(v)

    for k, v in dict_model_large.items():
        if 'attention.output.dense.weight' in k:  # 第0，2，4层参数，要固定，另外加入lora模块
            # print(k)
            # print(v)
            logger.info(k)
            logger.info(v)
    # 需要对old layer加入lora模块，同时要改变old layer的计算方式？要结合lora之后得到新的输出
    # 对所有layer都加入lora模块？那么forward函数也需要相应改变吧（只需要替换原始的Linear啥的？）？
    # 然后训练时，冻结1，3，5层的lora，运行0，2，4层的lora
    # 把大模型的每个layer，都变成loraLayer，然后根据name来冻住某些参数


    model_large.load_state_dict(dict_model_large)
    # ckpt['args'].encoder_layers = bm_layers
    # ckpt['args'].encoder_embed_dim = bm_hidden
    # ckpt['args'].encoder_attention_heads = bm_num_heads
    # ckpt['args'].encoder_ffn_embed_dim = bm_intermediate_size
    # ckpt['args'].arch = f'roberta_{bm_layers}layer_{bm_hidden}hidden_{bm_num_heads}head_{bm_intermediate_size}ffn'
    # ckpt['args'].layer_candidates = layer_candidates
    # ckpt['args'].layer_idxs = new_layer_idxs

    # torch.save(ckpt, save_path)

    return model_large


@torch.no_grad()  # 有必要吗？
def initialize_model_with_bert2bert(model_large, args):
    bm_layers = 6
    bm_hidden = 256
    bm_num_heads = 4
    bm_intermediate_size = 1024
    model_large = FPI(args, model_large, bm_layers, bm_hidden, bm_num_heads, bm_intermediate_size, add_noise=False, add_last=False, source_model=None)

    # lora_r = 8  # 改为256能不能逼近baseline结果？
    # lora_alpha = 1  # 16有点大吧？
    # # lora_dropout = 0.05  # 这个需要吗？
    # assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
    # # 对较大模型的old layer，将linear替换成lora
    # for i, layer in enumerate(model_large.bert.encoder.layer):
    #     if not i % 2:
    #         layer.attention.self.query = assign_lora(layer.attention.self.query)
    #         layer.attention.self.key = assign_lora(layer.attention.self.key)
    #         layer.attention.self.value = assign_lora(layer.attention.self.value)
    #         layer.attention.output.dense = assign_lora(layer.attention.output.dense)
    #         layer.intermediate.dense = assign_lora(layer.intermediate.dense)
    #         layer.output.dense = assign_lora(layer.output.dense)
    # logger.info('^^^^^^^^^^^^^^^^^^^^^^^^')
    # logger.info('^^^^^^^^^^^^^^^^^^^^^^^^')
    # logger.info('^^^^^^^^^^^^^^^^^^^^^^^^')
    # logger.info('^^^^^^^^^^^^^^^^^^^^^^^^')
    # logger.info('^^^^^^^^^^^^^^^^^^^^^^^^')
    # # logger.info(model_large)
    #
    # # 要把old layer的参数freeze掉？
    # for name, param in model_large.named_parameters():  # 要不要对embedding和cls加lora？要不要把layernorm排除在外？
    #     if ('0' in name or '2' in name or '4' in name) and 'lora' not in name and 'LayerNorm' not in name:
    #     # if '0' in name or '2' in name or '4' in name:  # 如果完全冻住old layer呢？包括lora也冻住？只训练new layers
    #         param.requires_grad = False
    #     logger.info(name)
    #     logger.info(param.requires_grad)

    return model_large


if __name__ == '__main__':
    # v = torch.randn(3, 3, 3)
    # choose_hidden_dims = get_choices(3, 3, is_always_left=False)
    # new_weight = wider3d(v, 0, 3, choices=[], div=True, add_noise=False)
    # # new_weight = wider2d(v, 0, 3, choices=[], div=True, add_noise=False)
    # # new_weight = wider(v, 8, choices=[], add_noise=False)
    # # print(new_weight.shape)
    # # print(v)
    # # print(new_weight)

    sm_layers = 3
    bm_layers = 6
    add_last = False
    layer_candidates = [i for i in range(sm_layers)]
    layer_idxs = [i for i in range(sm_layers)]
    added_layers = []
    added_layer_num = bm_layers - sm_layers
    print("number of added layer: " + str(added_layer_num))
    print("old layer candidates: " + str(layer_candidates))
    print("old layer idxs: " + str(layer_idxs))
    if len(layer_candidates) >= added_layer_num:  # 比如从6层grow到12层
        added_layers = layer_candidates[-added_layer_num:] if add_last else random.sample(layer_candidates,
                                                                                          added_layer_num)
        print('added_layers: ' + str(added_layers))
        for layer in added_layers:  # added_layers是
            layer_candidates.remove(layer)  # layer_candidates这不就为空了吗？
    sm_layer_idxs = [i for i in range(sm_layers)]
    sm_layer_idx_for_bert2bert_top = []
    new_layer_idxs = []
    for layer in sm_layer_idxs:
        idx = layer_idxs[layer]
        sm_layer_idx_for_bert2bert_top.append(layer)
        new_layer_idxs.append(idx)
        while idx in added_layers:
            sm_layer_idx_for_bert2bert_top.append(layer)
            new_layer_idxs.append(idx)
            added_layers.remove(idx)
    assert len(new_layer_idxs) == bm_layers
    assert len(new_layer_idxs) == len(sm_layer_idx_for_bert2bert_top)
    print("new layer candidates: " + str(layer_candidates))
    print("new layer idxs: " + str(new_layer_idxs))
    if len(new_layer_idxs) % (max(layer_idxs) + 1) == 0:
        new_layer_idxs = [i for i in range(bm_layers)]  # 又改成了大模型的实际index
        layer_candidates = [i for i in range(bm_layers)]

    print("final layer candidates: " + str(layer_candidates))
    print("final layer idxs: " + str(new_layer_idxs))
    print(
        "sm layer idx for bert2bert: " + str(sm_layer_idx_for_bert2bert_top))  # # 大模型的layer index，其中index都是小模型复制后的index



