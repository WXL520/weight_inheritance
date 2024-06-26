# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import argparse
import glob
import json
import logging
import os
import pickle
import random
import re
import shutil
import sys
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# from tqdm import tqdm, trange
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    WEIGHTS_NAME,
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from data import CoLDataset
from datasets import load_from_disk
from param import process_args
from model import SimpleBertForMaskedLM, SimpleRobertaForMaskedLM

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from bert2bert import initialize_model_with_bert2bert, FPI
from ligo import initialize_model_with_ligo

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, SimpleBertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, SimpleRobertaForMaskedLM, RobertaTokenizer),
}


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.col_data:  # 走这里，啥意思啊，按列处理数据吗？tokenize的过程吗？
        return CoLDataset(file_path, args.tokenizer_name, tokenizer, args.block_size,
                          split_sent=args.split_sent,
                          verbose=(args.gpu == 0))
    elif args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class DataCollatorNew(DataCollatorWithPadding):  # 这是干啥的？padding？
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        encoded_inputs = {key: [example[key] for example in features] for key in features[0].keys() if key != "labels"}
        batch = self.tokenizer.pad(
            encoded_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )  # 能不能不计算NSP loss啊？？？
        mlm_labels = []
        for example in features:
            exam_len = len(example["labels"])
            mlm_labels.append(example["labels"].copy() + [-100] * (self.max_length - exam_len))  # pad labels?

        batch["labels"] = torch.tensor(mlm_labels)  # 看看batch长什么样子？
        return batch


def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    set_seed(args)  # Added here for reproducibility

    """ Train the model """
    if args.gpu == 0:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        tb_writer = SummaryWriter(args.output_dir + '/runs/' + current_time)

    args.train_batch_size = args.per_gpu_train_batch_size

    # 这得改成MSG的collate函数吧？
    # def collate(examples: List[torch.Tensor]):
    #     if tokenizer._pad_token is None:
    #         return pad_sequence(examples, batch_first=True)
    #     return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Data collator
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    data_collator = DataCollatorNew(tokenizer=tokenizer, padding="max_length", max_length=max_seq_length)
    # DataLoaders creation:
    if args.shuffle:
        logger.info(f"Shuffle the dataset in training,"
                    f"GPU: {args.gpu},"
                    f"Rank: {args.rank},"
                    f"Total: {args.world_size}")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=args.shuffle,
    )
    train_dataloader = DataLoader(  # 这个dataloader的设置和MSG的有点不同啊？
        train_dataset, sampler=train_sampler, shuffle=False, num_workers=0,  # 为什么把shuffle设为false啊？
        batch_size=args.train_batch_size, collate_fn=data_collator, pin_memory=True
    )

    t_total = args.max_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    # logger.info("$$$"*300)
    # logger.info('^' * 200)
    # logger.info([n for n, p in model.named_parameters()])
    optimizer_grouped_parameters = [  # 为什么把params分两组？关于bias和LayerNorm.weight的参数，不使用weight_decay？
        {
            "params": [p for n, p in model.named_parameters() if n[19] in ['0', '2', '4'] and not any(nd in n for nd in no_decay)],  # n in old and not in decay
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n[19] in ['0', '2', '4'] and any(nd in n for nd in no_decay)],  # n in old and in decay
            "weight_decay": 0.0},

        {
            "params": [p for n, p in model.named_parameters() if n[19] in ['1', '3', '5'] and not any(nd in n for nd in no_decay)],  # n in new and not in decay
            "lr": args.learning_rate * 1.0,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n[19] in ['1', '3', '5'] and any(nd in n for nd in no_decay)],  # n in new and in decay
            "lr": args.learning_rate * 1.0,
            "weight_decay": 0.0},

    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      # betas=(0.9, 0.98),
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    if args.warmup_ratio > 0.:
        assert args.warmup_steps == 0
        args.warmup_steps = int(t_total * args.warmup_ratio)
    if args.gpu == 0:
        print("Optimized with lr %f, steps %d, warmup steps %d, and use beta, epsilon %0.8f." % (
            args.learning_rate, t_total, args.warmup_steps, optimizer.defaults['eps']
        ), optimizer.defaults['betas'])  # warmup 10000个steps是干嘛的？意思是前10000个steps没有学习率下降？
    if args.scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    elif args.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
            num_cycles=args.scheduler_cosine_cycles
        )
    elif args.scheduler_type == 'poly':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
            power=args.scheduler_poly_power
        )
    else:
        raise ValueError(f"Unknow lr scheduler: {args.scheduler_type}")

    # Check if saved optimizer or scheduler states exist
    if (  # nope
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"), map_location=torch.device('cpu')))
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"), map_location=torch.device('cpu')))

    if args.fp16:  # 后面尝试下这个？？？暂时不用
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level,
                                          verbosity=0)
        from apex.parallel import DistributedDataParallel as DDP
        model = DDP(model)
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))  # samples要和MSG相同才行
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(  # 哎呦，真实batch size还要乘上accumulation_steps？那每个gpu岂不是应该设置为16 batch size？？？问问tianyi
        # 好像只有Roberta的accumulation_steps被设为4了，bert就是1，那per gpu size设为64没毛病。。
        "  Total train batch size (w. distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * args.world_size
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)  # 数据量多了，可能要加大，当然模型小了，所以也不一定

    global_step = 0
    epochs_trained = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_name = os.path.basename(args.model_name_or_path)
            global_step = int(checkpoint_name.split("-")[-1])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from iter %d, epoch %d" % (global_step, epochs_trained))
        except ValueError:
            logger.info("  Do not load model from %s, restart training" % args.model_name_or_path)

    model.zero_grad()

    # IMPORTANT: save the initialization
    if args.gpu == 0 and global_step == 0:
        checkpoint_name = f"checkpoint-{global_step:08d}"
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        save_model(args, ckpt_dir, checkpoint_name, model, tokenizer, optimizer, scheduler)

    while True:
        if args.gpu == 0:
            logger.info(" Evaluation Results of step %d: " % global_step)
            results = evaluate(args, eval_dataset, model.module, tokenizer)
            # for key, value in results.items():
            #     tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            #     logger.info("\t %s: %0.4f" % (key, value))

            output_log_file = os.path.join(args.output_dir, "train_log.txt")
            with open(output_log_file, 'a') as f:
                eval_ppl = results['perplexity']
                print(
                    f"train_step={global_step}, lr={scheduler.get_lr()[0]},"
                    f"eval_ppl={eval_ppl}", file=f)
        # epoch_iterator = tqdm(train_dataloader, desc=f"Epoch: {epochs_trained:03d}", disable=args.gpu != 0)
        tr_loss, tr_lm_loss = 0.0, 0.0
        t_start = time.time()
        model.zero_grad()  # Support of accumulating gradients
        for step, batch in enumerate(train_dataloader):
            # 看看mask前后的数据分别长什么样子？
            # print('**' * 200)
            # print('**' * 200)
            # print(f'batch: {batch}')  # 原本的batch数据，是否包含102这个分隔符啊？
            # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)  # 应该提前把数据准备好，不要临时处理
            inputs = batch['input_ids']
            labels = batch['labels']
            attention_mask = batch['attention_mask']
            inputs = inputs.to(args.device)  # 这里的input好像只有128的长度，以及对应的label，很简洁。。
            labels = labels.to(args.device)
            attention_mask = attention_mask.to(args.device)
            # print(f'inputs: {inputs}')
            # print(f'labels: {labels}')
            # If some of the input is padded, then the attention mask is needed
            # attention_mask = (inputs != tokenizer.pad_token_id)  # word_tokens --> 1, pad_token --> 0
            # if attention_mask.all():  # 这里手动设置attention_mask?
            #     attention_mask = None

            model.train()
            outputs = model(inputs,
                            attention_mask=attention_mask,
                            masked_lm_labels=labels,
                            current_step=global_step) if args.mlm else model(inputs, labels=labels,
                                                                             current_step=global_step)
            loss = outputs['loss']  # model outputs are always tuple in transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps  # 对4个进程的loss取平均？

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            tr_lm_loss += outputs['lm_loss'].item()  # lm_loss是什么啊？
            if (step + 1) % args.gradient_accumulation_steps == 0:  # 每个进程都会给step加1？
                global_step += 1
                # if args.gpu == 0 and global_step % 1000 == 0:
                #     for name, param in model.named_parameters():
                #         # logger.info(name)
                #         if name == 'module.bert.encoder.layer.0.attention.self.key.weight':  # old
                #             # logger.info(torch.norm(param.grad))  # shape is [256,256]
                #             # logger.info(param.grad.shape)
                #             logger.info(" old param gradient norm ")
                #             logger.info(torch.norm(param.grad))
                #         elif name == 'module.bert.encoder.layer.1.attention.self.key.weight':  # new
                #             # logger.info(torch.norm(param.grad))  # shape is [256,256]
                #             # logger.info(param.grad.shape)
                #             logger.info(" new param gradient norm ")
                #             logger.info(torch.norm(param.grad))
                if args.max_grad_norm > 0.:
                    if args.fp16:
                        total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                # print('%' * 200)
                # print(f'lr : {scheduler.get_lr()}')

                if args.gpu == 0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    t_elapse = time.time() - t_start

                    # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    if args.fp16:
                        try:
                            from apex.amp import _amp_state
                            tb_writer.add_scalar("loss_scale", _amp_state.loss_scalers[0]._loss_scale, global_step)
                            tb_writer.add_scalar("scaled_loss", scaled_loss.item(), global_step)
                        except ImportError:
                            logger.warning("Cannot import apex.amp._amp_state, "
                                           "would not state the loss_scale in the log")
                    if args.max_grad_norm > 0.:  # Only clip the grad when it is valid
                        tb_writer.add_scalar("grad_norm", total_norm, global_step)
                    train_loss = tr_loss / args.logging_steps
                    train_ppl = torch.exp(torch.tensor(tr_lm_loss / args.logging_steps)).item()
                    tb_writer.add_scalar("loss", train_loss, global_step)
                    tb_writer.add_scalar("train_ppl", train_ppl, global_step)
                    tr_loss = tr_lm_loss = 0.

                    # also evaluate on valid set for ppl  仅在gpu0上执行eval？
                    logger.info(" Evaluation Results of step %d: " % global_step)
                    results = evaluate(args, eval_dataset, model.module, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        logger.info("\t %s: %0.4f" % (key, value))

                    output_log_file = os.path.join(args.output_dir, "train_log.txt")
                    with open(output_log_file, 'a') as f:
                        eval_ppl = results['perplexity']
                        print(
                            f"train_step={global_step}, train_time={t_elapse}, lr={scheduler.get_lr()[0]}, train_loss={train_loss},"
                            f"train_ppl={train_ppl}, eval_ppl={eval_ppl}", file=f)

                    t_start = time.time()

                # 哪里是最终的valid结果啊？参数继承的时候，怎么节省时间啊？
                # 应该就是训练完之后，去看log，啥时候能达到scratch的最好结果
                if args.gpu == 0 and args.ckpt_steps > 0 and global_step % args.ckpt_steps == 0:
                    checkpoint_name = f"checkpoint-{global_step:08d}"
                    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
                    os.makedirs(ckpt_dir, exist_ok=True)
                    save_model(args, ckpt_dir, checkpoint_name, model, tokenizer, optimizer, scheduler)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if args.max_steps > 0 and global_step >= args.max_steps:  # 终止条件如果是400k个steps，那么如何节省时间呢？
            # epoch_iterator.close()
            break

        epochs_trained += 1

    # consider during the last evaluation, the GPU 0 is still working while others have exited.
    # when GPU 0 call torch.no_grad, it will wait for the response from other processes
    # however, a deadlock will be caused if other processes just exit
    # torch.distributed.barrier()

    if args.gpu == 0:
        tb_writer.close()


def save_model(args, ckpt_dir, name, model, tokenizer, optimizer, scheduler):
    # Save model checkpoint
    output_dir = os.path.join(ckpt_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)


def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    data_collator = DataCollatorNew(tokenizer=tokenizer, padding="max_length", max_length=max_seq_length)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=data_collator
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    # for batch in tqdm(eval_dataloader, desc="Evaluating"):
    for i, batch in enumerate(eval_dataloader):
        # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)
        # If some of the input is padded, then the attention mask is needed
        # attention_mask = (inputs != tokenizer.pad_token_id)  # word_tokens --> 1, pad_token --> 0
        # if attention_mask.all():
        #     attention_mask = None

        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask, masked_lm_labels=labels) if args.mlm else model(
                inputs, labels=labels)
            lm_loss = outputs['lm_loss']
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss)).item()

    result = {"perplexity": perplexity}

    return result


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def main():
    parser = process_args()
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    port = 9595
    while is_port_in_use(port):
        port += 1
    print("Use port", port)
    os.environ['MASTER_PORT'] = str(port)

    # Using all available gpus for multi-processing distributed
    args.gpus = torch.cuda.device_count()
    print("Use gpus ", list(range(args.gpus)))
    args.world_size = args.gpus * args.nodes
    mp.spawn(setup, nprocs=args.gpus, args=(args,))  # nprocs个进程同时执行setup函数？


def setup(gpu, args):
    # gpu参数控制进程号
    if args.should_continue:  # 怎么还continue training呢？应该是仅在参数继承的时候用到吧
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        checkpoint_names = []
        if os.path.isdir(ckpt_dir):
            checkpoint_names = [fn for fn in os.listdir(ckpt_dir) if fn.startswith('checkpoint-')]
        if len(checkpoint_names) > 0:
            checkpoint_names = sorted(checkpoint_names, key=lambda p: int(p.split('-')[-1]))
            args.model_name_or_path = os.path.join(ckpt_dir, checkpoint_names[-1])
        else:
            logger.warning('No checkpoint detected: %s', ckpt_dir)  # 4个gpu进程都打印了这个信息。。
            args.model_name_or_path = None

    # Setup CUDA, GPU & distributed training
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    args.gpu = gpu  # Local device id.
    args.device = device  # Local device object.
    args.rank = args.nr * args.gpus + gpu  # The gpu id in the world.
    torch.distributed.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(seconds=1800)  # 加上这个解决了nccl卡死问题
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.gpu == 0 else logging.WARN,
    )
    logger.warning(
        "Process GPU: %s, num_of_total_GPUs: %s, distributed training: True, 16-bits training: %s",  # 16-bits是false？
        args.gpu, args.gpus, args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and token
    # Barrier to make sure only the first process in distributed training
    # download model & vocabizer
    if gpu != 0:
        torch.distributed.barrier()  # 只允许gpu0来加载模型和vocab？

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Get Config
    if args.config_name:  # 这里加载param.py里面的通用config，也加载了robeta_wiki.txt里面的默认参数吗？
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "Why do you want the default config?? Please use --config_name or --model_name_or_path"
        )

    # Get Data
    args.dataset_name = '../MSG-master/bert_data/static_10000'
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        data = load_from_disk("{}".format(args.dataset_name))  # 这个数据怎么处理？
    else:
        raise ValueError(
            "Should load a static masked dataset."
        )
    # Preprocessing of the datasets is done outside.
    train_dataset = data["train"]
    eval_dataset = data["test"]

    # Get Tokenizer
    # if args.tokenizer_name:  # 这里的cache_dir为啥为空啊？那继承的是什么tokenizer啊？
    #     tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    #     # BERT always needs lower cased tokens.
    #     if 'uncased' in args.model_type:
    #         assert tokenizer.init_kwargs.get("do_lower_case", False)
    # elif args.model_name_or_path:
    #     tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    # else:
    #     raise ValueError(
    #         "You are instantiating a new {} tokenizer. This is not supported, "
    #         "but you can do it from another script, save it,"
    #         "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
    #     )
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, cache_dir="../MSG-master/bert_cache")
    elif args.model_name_or_path:  # 加载保存的tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, cache_dir="../MSG-master/bert_cache")
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    assert args.block_size <= tokenizer.model_max_length

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            args=args
        )
    elif args.source_model_path or args.pretrained_ligo_path:
        logger.info("Growing [%s] new model from: %s", args.grow_scheme, args.source_model_path)

        model = model_class(config=config, args=args)  # 在这里设置bert的层数等超参？

        if args.grow_scheme == 'none':
            logger.info("No initialization scheme applied. Training new model with random initialization ...")
        elif args.grow_scheme == 'ligo':
            ckpt_dir = os.path.join(args.pretrained_ligo_path, 'checkpoints')
            checkpoint_names = [fn for fn in os.listdir(ckpt_dir) if fn.startswith('checkpoint-')]
            checkpoint_names = sorted(checkpoint_names, key=lambda p: int(p.split('-')[-1]))
            args.pretrained_ligo_path = os.path.join(ckpt_dir, checkpoint_names[-1])

            args.fuse_init_scheme_depth = args.fuse_init_scheme_width = args.fuse_init_scheme[0]
            if len(args.fuse_init_scheme) >= 2:
                args.fuse_init_scheme_width = args.fuse_init_scheme[1]
            args.fuse_init_noise_depth = args.fuse_init_noise_width = args.fuse_init_noise[0]
            if len(args.fuse_init_noise) >= 2:
                args.fuse_init_noise_width = args.fuse_init_noise[1]

            model = initialize_model_with_ligo(model, args)
        elif args.grow_scheme == 'bert2bert':
            logger.info("#################################")
            logger.info("#################################")
            logger.info("#################################")
            logger.info("#################################")
            logger.info("#################################")
            logger.info("Initialize model from bert2bert")
            model = initialize_model_with_bert2bert(model, args)
        else:
            raise NotImplementedError(f'Grow method [{args.grow_scheme}] not implemented yet!')

    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config, args=args)  # model不用改吧？
        # bert2bert相当于train from scratch？只是初始化参数要单独加载一下？需要从旧参数copy？在这里给参数赋值？

    # model.resize_token_embeddings(len(tokenizer))  # 也没用添加token啊，为啥加上这个？要不要加？
    model.to(args.device)

    # End of barrier to make sure only the first process waiting other processes
    if gpu == 0:
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # Barrier to make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        # if gpu != 0:
        #     torch.distributed.barrier()
        # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        # if gpu == 0:
        #     torch.distributed.barrier()  # 这不是把所有进程都barrier掉了吗？

        train(args, train_dataset, eval_dataset, model, tokenizer)

    # Evaluation
    if args.do_eval and gpu == 0:
        result = evaluate(args, eval_dataset, model, tokenizer)


if __name__ == "__main__":
    main()
