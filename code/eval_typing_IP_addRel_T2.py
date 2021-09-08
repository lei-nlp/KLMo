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
import simplejson as json
import codecs

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

#from knowledge_bert.typing_IP import BertTokenizer as BertTokenizer_label
from knowledge_bert.tokenization_IP_addRel import BertTokenizer
from knowledge_bert.ori_modeling_addRelation_T2 import BertForSequenceClassification
from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# read entity mapping
entity2id = {}
with open("/apdcephfs/common/bettyleihe/Pretrained_Encyclopedia/kg_embed_CH/entity2id.txt", 'r') as fin:
    fin.readline()
    while 1:
        l = fin.readline()
        if l == "":
            break
        ent, idx = l.strip().split()
        entity2id[ent] = int(idx)
print (len(entity2id))


# read all triples
ee2rels = {}
for line in open("/dockerdata/bettyleihe/Pretrained_Encyclopedia/kg_embed_CH/all_triples", 'r', encoding='UTF-8'):
    parts = line.strip().split('\t')
    if len(parts) != 3:
        continue
    he = parts[0].strip()
    te = parts[1].strip()
    r = int(parts[2].strip())
    r = r + 2466069  # relation index from 2466069 after all ents
    ee = he+'_'+te
    if ee in ee2rels:
        if r not in ee2rels[ee]:
            ee2rels[ee].append(r)
    else:
        ee2rels[ee] = [r]
print (len(ee2rels))

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, left_pos, right_pos, ent_ids, er_ids, er_mask, er_segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        #self.input_ent = input_ent
        #self.ent_mask = ent_mask
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.ent_ids =  ent_ids
        self.er_ids = er_ids
        self.er_mask = er_mask
        self.er_segment_ids = er_segment_ids



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        lines = []
        for line in codecs.open(input_file,'r','utf8'):
            line = line.strip()
            lines.append(line)
        return lines


class TypingProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "new.15k.conll.json")))
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "new.15k.conll.json")), "train")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "new.8k.conll.json")), "test")


    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line_) in enumerate(lines):
            line = json.loads(line_)
            guid = i
            text_a = (line['sent'], [["SPAN", line["start"], line["end"]]])
            text_b = line['ents']
            #label = line['lables'][0]
            label = line['new_labels'][-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def gen_ER_list(eid_list):
    er_list = []
    for i in range(len(eid_list)):
        for j in range(i+1, len(eid_list)):
            key_str = str(eid_list[i])+"_"+str(eid_list[j])
            if key_str in ee2rels:
                rl = ee2rels[key_str]
                for r in rl:
                    er_list.append(r)
                    er_list.append(eid_list[i])
                    er_list.append(eid_list[j])
            key_str = str(eid_list[j])+"_"+str(eid_list[i])
            if key_str in ee2rels:
                rl = ee2rels[key_str]
                for r in rl:
                    er_list.append(r)
                    er_list.append(eid_list[j])
                    er_list.append(eid_list[i])
    return er_list
    
def create_ent_predictions(tokens, entity):
    ent_left_indexes = []
    ent_right_indexes = []
    ent_ids = []
    ent_uniq_ids = []
    for (i, ent) in enumerate(entity):
        if i > 0 and entity[i] != '-1' and entity[i] in entity2id:
            if entity[i-1] == '-1' or entity[i-1] != entity[i]:
                ent_left_indexes.append(i-1)
                eid = entity2id[entity[i]]
                ent_ids.append(eid)
                if eid not in ent_uniq_ids:
                    ent_uniq_ids.append(eid)
        if i < len(entity)-1 and entity[i] != '-1' and entity[i] in entity2id:
            if entity[i+1] == '-1' or entity[i+1] != entity[i]:
                ent_right_indexes.append(i+1)
    assert len(ent_left_indexes) == len(ent_right_indexes)
    assert len(ent_ids) == len(ent_right_indexes)
    return (ent_left_indexes, ent_right_indexes, ent_ids, ent_uniq_ids)
    

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_entity_per_seq, max_relation_per_seq, threshold):
    """Loads a data file into a list of `InputBatch`s."""
    label_list = sorted(label_list)
    label_map = {label : i for i, label in enumerate(label_list)}
    '''
    entity2id = {}
    with open("kg_embed_CH/entity2id.txt") as fin:
        fin.readline()
        for line in fin:
            qid, eid = line.strip().split('\t')
            entity2id[qid] = int(eid)
    '''
    
    features = []
    for (ex_index, example) in enumerate(examples):
        ex_text_a = example.text_a[0]
        ex_text_a_toks = ex_text_a.split(' ')
        h = example.text_a[1][0]
        ex_text_a_toks = ex_text_a_toks[:h[1]] + ["[unused99]"] + ex_text_a_toks[h[1]:h[2]] + ["[unused99]"] + ex_text_a_toks[h[2]:]
        ex_text_a = ' '.join(ex_text_a_toks)
        #begin, end = h[1:3]
        #h[1] += 1
        #h[2] += 1
        #tokens_a, entities_a = tokenizer_label.tokenize(ex_text_a, [h])
        # change begin pos
        ent_pos = [x for x in example.text_b] #if x[4]>threshold]
        for x in ent_pos:
            cnt = 0
            if x[2] > h[2]:
                cnt += 1
            if x[2] >= h[1]:
                cnt += 1
            x[2] += cnt
            x[3] += cnt
            
        tokens_a, entities_a = tokenizer.tokenize(ex_text_a, ent_pos)
        if h[1] == h[2]:
            continue
        #mark = False
        tokens_b = None
        #for e in entities_a:
        #    if e != "UNK":
        #        mark = True
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            entities_a = entities_a[:(max_seq_length - 2)]
            #entities = entities[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        ents = ["-1"] + entities_a + ["-1"]
        #real_ents = ["UNK"] + entities + ["UNK"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        (ent_before_positions, ent_after_positions, ent_ids, ent_uniq_ids) = create_ent_predictions(input_ids, ents)
        er_ids = gen_ER_list(ent_uniq_ids)  ## generate entity+relation input: r1 e1 e2 r2 e1 e3 ...
        
        '''
        span_mask = []
        for ent in ents:
            if ent != "UNK":
                span_mask.append(1)
            else:
                span_mask.append(0)

        input_ent = []
        ent_mask = []
        for ent in real_ents:
            if ent != "UNK" and ent in entity2id:
                input_ent.append(entity2id[ent])
                ent_mask.append(1)
            else:
                input_ent.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1

        if not mark:
            print(example.guid)
            print(example.text_a[0])
            print(example.text_a[0][example.text_a[1][0][1]:example.text_a[1][0][2]])
            print(ents)
            exit(1)
        if sum(span_mask) == 0:
            continue
        '''
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_ = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        #ent_mask += padding
        #input_ent += padding_
        
        if len(ent_before_positions) >= max_entity_per_seq:
            ent_before_positions = ent_before_positions[:max_entity_per_seq]
            ent_after_positions = ent_after_positions[:max_entity_per_seq]
            ent_ids = ent_ids[:max_entity_per_seq]
        else:
            rest = max_entity_per_seq - len(ent_before_positions)
            ent_before_positions.extend([0]*rest)
            ent_after_positions.extend([0]*rest)
            ent_ids.extend([-1]*rest)
            
        max_er_len = max_relation_per_seq * 3
        er_mask = [1] * len(er_ids)
        er_segments = [1] * len(er_ids)
        if len(er_ids) >= max_er_len:
            er_ids = er_ids[:max_er_len]
            er_mask = er_mask[:max_er_len]
            er_segments = er_segments[:max_er_len]
        else:
            rest = max_er_len - len(er_ids)
            er_ids.extend([-1]*rest)
            er_mask.extend([0]*rest)
            er_segments.extend([1]*rest)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        #assert len(ent_mask) == max_seq_length
        #assert len(input_ent) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("Entity: %s" % example.text_a[1])
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in zip(tokens, ents)]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info(ents)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              left_pos = ent_before_positions, 
                              right_pos = ent_after_positions, 
                              ent_ids = ent_ids, 
                              er_ids = er_ids, 
                              er_mask = er_mask, 
                              er_segment_ids = er_segments,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, ents_a, ents_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            ents_a.pop()
        else:
            tokens_b.pop()
            ents_b.pop()
'''
def accuracy(out, l):
    cnt = 0
    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        top = max(x1)
        for i in range(len(x1)):
            if x1[i] > 0:
                yy1.append(i)
            if x2[i] > 0:
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
        cnt += set(yy1) == set(yy2)
    return cnt, y1, y2
'''

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
    

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def eval_result(pred_result, labels, na_id):
    correct = 0
    total = len(labels)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    for i in range(total):
        if labels[i] == pred_result[i]:
            correct += 1
            if labels[i] != na_id:
                correct_positive += 1
        if labels[i] != na_id:
            gold_positive += 1
        if pred_result[i] != na_id:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    result = {'acc': acc, 'micro_p': micro_p,
              'micro_r': micro_r, 'micro_f1': micro_f1}
    return result
    
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--ernie_model", default=None, type=str, required=True,
                        help="Ernie pre-trained model")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
                             
    parser.add_argument("--max_entity_per_seq", default=20,type=int,
                        help="Maximum number of entities per sequence.")
    parser.add_argument("--max_relation_per_seq", default=30,type=int,
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
                        default=128,
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
    parser.add_argument('--threshold', type=float, default=.3)

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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    processor = TypingProcessor()

    #tokenizer_label = BertTokenizer_label.from_pretrained(args.ernie_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained(args.ernie_model, do_lower_case=args.do_lower_case)

    train_examples, label_list = processor.get_train_examples(args.data_dir)
    num_labels = len(label_list)
    label_list = sorted(label_list)
    #class_weight = [min(d[x], 100) for x in label_list]
    #logger.info(class_weight)
    
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

    filenames = os.listdir(args.output_dir)
    filenames = [x for x in filenames if "pytorch_model.bin_" in x]

    file_mark = []
    for x in filenames:
        #file_mark.append([x, True])
        file_mark.append([x, False])

    eval_examples = processor.get_test_examples(args.data_dir)
    test = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, args.max_entity_per_seq, args.max_relation_per_seq, args.threshold)

    
    for x, mark in file_mark:
        #print(x, mark)
        output_model_file = os.path.join(args.output_dir, x)
        model_state_dict = torch.load(output_model_file)
        model, _ = BertForSequenceClassification.from_pretrained(args.ernie_model, state_dict=model_state_dict, num_labels=len(label_list))
        model.to(device)
        
        '''
        if mark:
            eval_examples = processor.get_dev_examples(args.data_dir)
        else:
            eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer_label, tokenizer, args.threshold)
        '''
        if mark == False:
            eval_features = test
            output_file = os.path.join(
                args.output_dir, "test_pred_{}.txt".format(x.split("_")[-1]))
            output_file_ = os.path.join(
                args.output_dir, "test_gold_{}.txt".format(x.split("_")[-1]))
        fpred = open(output_file, "w")
        fgold = open(output_file_, "w")
        
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        # zeros = [0 for _ in range(args.max_seq_length)]
        # zeros_ent = [0 for _ in range(100)]
        # zeros_ent = [zeros_ent for _ in range(args.max_seq_length)]
        
        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long)
        #all_ent = torch.tensor(
        #    [f.input_ent for f in eval_features], dtype=torch.long)
        #all_ent_masks = torch.tensor(
        #    [f.ent_mask for f in eval_features], dtype=torch.long)
        
        all_left_pos = torch.tensor([f.left_pos for f in eval_features], dtype=torch.long)
        all_right_pos = torch.tensor([f.right_pos for f in eval_features], dtype=torch.long)
        all_ent_ids = torch.tensor([f.ent_ids for f in eval_features], dtype=torch.long)
        all_er_ids = torch.tensor([f.er_ids for f in eval_features], dtype=torch.long)
        all_er_mask = torch.tensor([f.er_mask for f in eval_features], dtype=torch.long)
        all_er_segment_ids = torch.tensor([f.er_segment_ids for f in eval_features], dtype=torch.long)
        
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_left_pos, all_right_pos, all_ent_ids, all_er_ids, all_er_mask, all_er_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        pred_all, label_all = [], []
        for input_ids, input_mask, segment_ids, left_pos, right_pos, ent_ids, er_ids, er_mask, er_segment_ids, label_ids in eval_dataloader:
            er_ids = embed(er_ids+1)  # -1 -> 0
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            left_pos = left_pos.to(device)
            right_pos = right_pos.to(device)
            ent_ids = ent_ids.to(device)
            er_ids = er_ids.to(device)
            er_mask = er_mask.to(device)
            er_segment_ids = er_segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, left_pos, right_pos, ent_ids, er_ids, er_mask, er_segment_ids, label_ids)
                    #input_ids, segment_ids, input_mask, input_ent, ent_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask, left_pos, right_pos, ent_ids, er_ids, er_mask, er_segment_ids)
                               #input_ids, segment_ids, input_mask, input_ent, ent_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            pred = np.argmax(logits, axis=1)
            for a, b in zip(pred, label_ids):
                pred_all.append(a)
                label_all.append(b)
                fgold.write("{}\n".format(label_list[b]))
                fpred.write("{}\n".format(label_list[a]))

            eval_loss += tmp_eval_loss.mean().item()
            

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        #result = eval_result(pred_all, label_all, label_list.index("IP"))
        result = eval_result(pred_all, label_all, -1)


        if mark == False:
            output_eval_file = os.path.join(
                args.output_dir, "test_results_{}.txt".format(x.split("_")[-1]))

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
