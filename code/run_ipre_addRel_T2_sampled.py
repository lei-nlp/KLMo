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

from knowledge_bert.tokenization_RC_addRel import BertTokenizer
from knowledge_bert.ori_modeling_addRelation_T2 import BertForSequenceClassification
from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

rng = random.Random(42)

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
        #with open(input_file, "r", encoding='utf-8') as f:
        #    return json.loads(f.read())
        lines = []
        for line in codecs.open(input_file,'r','utf8'):
            line = line.strip()
            lines.append(line)
        return lines
        
class TacredProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")
    
    def get_labels(self):
        """Useless"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line_) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line = json.loads(line_)
            text_a = (line['text'], line['ents'])
            label = line['label']
            #line['ann'] = []
            if label == 'NA' and rng.random() > 0.7:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=line['ann'], label=label))
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
        '''
        if i > 0 and entity[i] != '-1' and entity[i-1] == '-1' and entity[i] in entity2id:
            ent_left_indexes.append(i-1)
            eid = entity2id[entity[i]]
            ent_ids.append(eid)
            if eid not in ent_uniq_ids:
                ent_uniq_ids.append(eid)
        if i < len(entity)-1 and entity[i] != '-1' and entity[i+1] == '-1' and entity[i] in entity2id:
            ent_right_indexes.append(i+1)
        '''
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
        h, t = example.text_a[1]
        #h_name = ex_text_a[h[1]:h[2]]
        #t_name = ex_text_a[t[1]:t[2]]
        #ex_text_a = ex_text_a.replace(h_name, "# "+h_name+" #", 1)
        #ex_text_a = ex_text_a.replace(t_name, "$ "+t_name+" $", 1)
        # Add [HD] and [TL], which are "#" and "$" respectively.
        if h[1] < t[1]:
            #ex_text_a = ex_text_a[:h[1]] + "# "+h_name+" #" + ex_text_a[h[2]:t[1]] + "$ "+t_name+" $" + ex_text_a[t[2]:]
            ex_text_a_toks = ex_text_a_toks[:h[1]] + ["[unused98]"] + ex_text_a_toks[h[1]:h[2]] + ["[unused98]"] + ex_text_a_toks[h[2]:t[1]] + ["[unused99]"] + ex_text_a_toks[t[1]:t[2]] + ["[unused99]"] + ex_text_a_toks[t[2]:]
        else:
            #ex_text_a = ex_text_a[:t[1]] + "$ "+t_name+" $" + ex_text_a[t[2]:h[1]] + "# "+h_name+" #" + ex_text_a[h[2]:]
            ex_text_a_toks = ex_text_a_toks[:t[1]] + ["[unused99]"] + ex_text_a_toks[t[1]:t[2]] + ["[unused99]"] + ex_text_a_toks[t[2]:h[1]] + ["[unused98]"] + ex_text_a_toks[h[1]:h[2]] + ["[unused98]"] + ex_text_a_toks[h[2]:]
        

        ent_pos = [x for x in example.text_b] #if x[-1]>threshold]
        for x in ent_pos:
            cnt = 0
            if x[2] > h[2]:
                cnt += 1
            if x[2] >= h[1]:
                cnt += 1
            if x[2] >= t[1]:
                cnt += 1
            if x[2] > t[2]:
                cnt += 1
            x[2] += cnt
            x[3] += cnt
        
        ex_text_a = ' '.join(ex_text_a_toks)
        tokens_a, entities_a = tokenizer.tokenize(ex_text_a, ent_pos)
        
        tokens_b = None
        if False:
            tokens_b, entities_b = tokenizer.tokenize(example.text_b[0], [x for x in example.text_b[1] if x[-1]>threshold])
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, entities_a, entities_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                entities_a = entities_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        ents = ["-1"] + entities_a + ["-1"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            ents += entities_b + ["-1"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        (ent_before_positions, ent_after_positions, ent_ids, ent_uniq_ids) = create_ent_predictions(input_ids, ents)
        er_ids = gen_ER_list(ent_uniq_ids)  ## generate entity+relation input: r1 e1 e2 r2 e1 e3 ...
        
        '''
        input_ent = []
        ent_mask = []
        for ent in ents:
            if ent != "-1" and ent in entity2id:
                input_ent.append(entity2id[ent])
                ent_mask.append(1)
            else:
                input_ent.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1
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
        #input_ent += padding_
        #ent_mask += padding
        
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
        #assert len(input_ent) == max_seq_length
        #assert len(ent_mask) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("ents: %s" % " ".join(
                    [str(x) for x in ents]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

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

    processors = TacredProcessor

    num_labels_task = 35

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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)


    processor = processors()
    label_list = None

    tokenizer = BertTokenizer.from_pretrained(args.ernie_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    train_examples, label_list = processor.get_train_examples(args.data_dir)
    num_labels = len(label_list)
    
    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model, _ = BertForSequenceClassification.from_pretrained(args.ernie_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
              num_labels = num_labels)
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
    no_grad = ['bert.encoder.layer.11.output.dense_ent', 'bert.encoder.layer.11.output.LayerNorm_ent']
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_grad)]
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
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)
                              # max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, args.max_entity_per_seq, args.max_relation_per_seq, args.threshold)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        
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

        # zeros = [0 for _ in range(args.max_seq_length)]
        # zeros_ent = [0 for _ in range(100)]
        # zeros_ent = [zeros_ent for _ in range(args.max_seq_length)]
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        #all_ent = torch.tensor([f.input_ent for f in train_features], dtype=torch.long)
        #all_ent_masks = torch.tensor([f.ent_mask for f in train_features], dtype=torch.long)
        all_left_pos = torch.tensor([f.left_pos for f in train_features], dtype=torch.long)
        all_right_pos = torch.tensor([f.right_pos for f in train_features], dtype=torch.long)
        all_ent_ids = torch.tensor([f.ent_ids for f in train_features], dtype=torch.long)
        all_er_ids = torch.tensor([f.er_ids for f in train_features], dtype=torch.long)
        all_er_mask = torch.tensor([f.er_mask for f in train_features], dtype=torch.long)
        all_er_segment_ids = torch.tensor([f.er_segment_ids for f in train_features], dtype=torch.long)
        
                              
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_left_pos, all_right_pos, all_ent_ids, all_er_ids, all_er_mask, all_er_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        output_loss_file = os.path.join(args.output_dir, "loss")
        loss_fout = open(output_loss_file, 'w')
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) if i != 6 else t for i, t in enumerate(batch))
                #input_ids, input_mask, segment_ids, , er_ids, er_mask, er_segment_ids, label_ids = batch
                input_ids, input_mask, segment_ids, left_pos, right_pos, ent_ids, er_ids, er_mask, er_segment_ids, label_ids = batch
                er_ids = embed(er_ids+1).to(device) # -1 -> 0
                loss = model(input_ids, segment_ids, input_mask, left_pos, right_pos, ent_ids, er_ids.half(), er_mask, er_segment_ids, label_ids)
                #loss = model(input_ids, segment_ids, input_mask, left_pos, right_pos, ent_ids, er_ids, er_mask, er_segment_ids, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                loss_fout.write("{}\n".format(loss.item()))
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
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(epoch))
            torch.save(model_to_save.state_dict(), output_model_file)

        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

if __name__ == "__main__":
    main()
