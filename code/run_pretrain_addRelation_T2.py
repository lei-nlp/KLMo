# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import codecs

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from knowledge_bert.tokenization import BertTokenizer
from knowledge_bert.ori_modeling_addRelation_T2 import BertForPreTraining
from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ent_per_seq", default=20, type=int,
                        help="Maximum number of entities per sequence.")
    parser.add_argument("--max_relation_per_seq", default=30, type=int,
                        help="Maximum number of relations per sequence.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    vecs = []
    vecs.append([0]*100) # CLS
    with open("kg_embed_CH/vec.d100", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed = torch.FloatTensor(vecs)
    embed = torch.nn.Embedding.from_pretrained(embed)
    #embed = torch.nn.Embedding(5041175, 100)

    logger.info("Shape of entity embedding: "+str(embed.weight.size()))
    del vecs
    
    train_data = None
    num_train_steps = None
    
    '''
    # load label dict
    ent_type_2_id = {}
    for line in codecs.open('/dockerdata/bettyleihe/Pretrained_Encyclopedia/fusion_info/type2id.dict','r','utf8'):
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        ent_type = int(parts[0].strip())
        type_id = int(parts[1].strip())
        ent_type_2_id[ent_type] = type_id 
    '''
    
    if args.do_train:
        # TODO
        import indexed_dataset
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,BatchSampler
        import iterators
        train_data = indexed_dataset.IndexedDataset(args.data_dir, fix_lua_indexing=True)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_sampler = BatchSampler(train_sampler, args.train_batch_size, True)
        def collate_fn(x):
            x = torch.LongTensor([xx for xx in x])
            input_ids, input_mask, segment_ids, masked_lm_labels = x[:,:args.max_seq_length], x[:,args.max_seq_length:2*args.max_seq_length], x[:,2*args.max_seq_length:3*args.max_seq_length], x[:,3*args.max_seq_length:4*args.max_seq_length]
            ent_before_positions = x[:,4*args.max_seq_length:4*args.max_seq_length+args.max_ent_per_seq]
            ent_after_positions = x[:,4*args.max_seq_length+args.max_ent_per_seq:4*args.max_seq_length+2*args.max_ent_per_seq]
            ent_ids = x[:,4*args.max_seq_length+2*args.max_ent_per_seq:4*args.max_seq_length+3*args.max_ent_per_seq]
            er_max_len = args.max_relation_per_seq*3
            er_ids = x[:,4*args.max_seq_length+3*args.max_ent_per_seq:4*args.max_seq_length+3*args.max_ent_per_seq+er_max_len]
            er_mask = x[:,4*args.max_seq_length+3*args.max_ent_per_seq+er_max_len:4*args.max_seq_length+3*args.max_ent_per_seq+2*er_max_len]
            er_segments = x[:,4*args.max_seq_length+3*args.max_ent_per_seq+2*er_max_len:4*args.max_seq_length+3*args.max_ent_per_seq+3*er_max_len]
            # build candidate
            uniq_idx = np.unique(ent_ids.numpy())  # may contain -1
            ent_candidate = embed(torch.LongTensor(uniq_idx+1))
            ent_candidate = ent_candidate.repeat([n_gpu, 1])
            # build entity labels
            d = {}
            for i, idx in enumerate(uniq_idx):
                d[idx] = i
            ent_labels = ent_ids.clone()
            d[-1] = -1
            ent_labels = ent_labels.apply_(lambda x: d[x])
            
            # build er input
            er_emb = embed(er_ids+1)
            return input_ids, input_mask, segment_ids, masked_lm_labels, ent_before_positions, ent_after_positions, ent_candidate, ent_labels, er_emb, er_mask, er_segments
        train_iterator = iterators.EpochBatchIterator(train_data, collate_fn, train_sampler)
        num_train_steps = int(
            len(train_data) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model, missing_keys = BertForPreTraining.from_pretrained(args.bert_model,
                   cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    #no_linear = ['layer.2.output.dense_ent', 'layer.2.intermediate.dense_1', 'bert.encoder.layer.2.intermediate.dense_1_ent', 'layer.2.output.LayerNorm_ent']
    #no_linear = [x.replace('2', '11') for x in no_linear]
    #param_optimizer = [(n, p) for n, p in param_optimizer if not any(nl in n for nl in no_linear)]
    #param_optimizer = [(n, p) for n, p in param_optimizer if not any(nl in n for nl in missing_keys)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            #from apex.optimizers import FP16_Optimizer
            from apex.fp16_utils.fp16_optimizer import FP16_Optimizer
            #from apex.contrib.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
            #from apex.contrib.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)
                             #  max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        #logger.info(dir(optimizer))
        #op_path = os.path.join(args.bert_model, "pytorch_op.bin")
        #optimizer.load_state_dict(torch.load(op_path))

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        model.train()
        import datetime
        fout = open(os.path.join(args.output_dir, "loss.{}".format(datetime.datetime.now())), 'w')
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_iterator.next_epoch_itr(), desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, masked_lm_labels, ent_left_positions, ent_right_positions, ent_candidate, ent_labels, er_emb, er_mask, er_segments = batch
                if args.fp16:
                    loss, masked_lm_loss, ent_loss = model(input_ids, segment_ids, input_mask, masked_lm_labels, ent_left_positions, ent_right_positions, ent_candidate.half(), ent_labels, er_emb.half(), er_mask, er_segments)
                else:
                    loss, masked_lm_loss, ent_loss = model(input_ids, segment_ids, input_mask, masked_lm_labels, ent_left_positions, ent_right_positions, ent_candidate, ent_labels, er_emb, er_mask, er_segments)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                    masked_lm_loss = masked_lm_loss.mean()
                    ent_loss = ent_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                fout.write("{} {} {}\n".format(loss.item()*args.gradient_accumulation_steps, masked_lm_loss.item(), ent_loss.item()))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if global_step % 10000 == 0 or global_step % 25000 == 0:
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(global_step))
                        torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(epoch))
            torch.save(model_to_save.state_dict(), output_model_file)
        fout.close()

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

if __name__ == "__main__":
    main()

