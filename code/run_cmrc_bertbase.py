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

import random
import os
import argparse
import numpy as np
import json
import torch
import utils
#from knowledge_bert.pytorch_modeling import ALBertConfig, ALBertForQA
from knowledge_bert.ori_modeling import BertConfig, BertForQuestionAnswering
from optimizations.pytorch_optimization import get_optimization, warmup_linear
from cmrc_evaluate.cmrc2018_output import write_predictions
from cmrc_evaluate.cmrc2018_evaluate import get_eval
import collections
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from tokenizations import offical_tokenization as tokenization
from preprocess.cmrc2018_preprocess import json2features

def evaluate(model, args, eval_examples, eval_features, device, global_steps, best_f1, best_em, best_f1_em):
    print("***** Eval *****")
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    output_prediction_file = os.path.join(args.output_dir,
                                          "predictions_steps" + str(global_steps) + ".json")
    output_nbest_file = output_prediction_file.replace('predictions', 'nbest')

    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.n_batch, shuffle=False)

    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    write_predictions(eval_examples, eval_features, all_results,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)

    tmp_result = get_eval(args.dev_file, output_prediction_file)
    tmp_result['STEP'] = global_steps
    log_file = args.output_dir+"/log.txt"
    with open(log_file, 'a') as aw:
        aw.write(json.dumps(tmp_result) + '\n')
    print(tmp_result)

    if float(tmp_result['F1']) > best_f1:
        best_f1 = float(tmp_result['F1'])

    if float(tmp_result['EM']) > best_em:
        best_em = float(tmp_result['EM'])

    if float(tmp_result['F1']) + float(tmp_result['EM']) > best_f1_em:
        best_f1_em = float(tmp_result['F1']) + float(tmp_result['EM'])
        utils.torch_save_model(model, args.output_dir,
                               {'f1': float(tmp_result['F1']), 'em': float(tmp_result['EM'])}, max_save_num=1)

    model.train()

    return best_f1, best_em, best_f1_em
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')

    # training parameter
    parser.add_argument('--train_epochs', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--float16', type=bool, default=False)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--eval_epochs', type=float, default=0.5)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--vocab_size', type=int, default=21128)

    # data dir
    parser.add_argument('--train_dir', type=str,
                        default='data/cmrc2018/train_features.json')
    parser.add_argument('--dev_dir1', type=str,
                        default='data/cmrc2018/dev_examples.json')
    parser.add_argument('--dev_dir2', type=str,
                        default='data/cmrc2018/dev_features.json')
    parser.add_argument('--train_file', type=str,
                        default='data/cmrc2018/cmrc2018_train.json')
    parser.add_argument('--dev_file', type=str,
                        default='data/cmrc2018/cmrc2018_dev.json')
    parser.add_argument("--bert_model", default="bert_base_chinese", type=str,
                        help="pre-trained model")
    parser.add_argument("--output_dir", default="output_cmrc_bertbase", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    #parser.add_argument('--bert_config_file', type=str,
    #                    default='check_points/pretrain_models/albert_large_zh/albert_config_large.json')
    #parser.add_argument('--vocab_file', type=str,
    #                    default='check_points/pretrain_models/albert_large_zh/vocab.txt')
    #parser.add_argument('--init_restore_dir', type=str,
    #                    default='check_points/pretrain_models/albert_large_zh/pytorch_albert_model.pth')
    #parser.add_argument('--checkpoint_dir', type=str,
    #                    default='check_points/cmrc2018/albert_large_zh/')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    #parser.add_argument('--log_file', type=str, default='log.txt')
    
    
    # use some global vars for convenience
    args = parser.parse_args()
    args.output_dir += ('/epoch{}_batch{}_lr{}_warmup{}_anslen{}/'
                            .format(args.train_epochs, args.n_batch, args.lr, args.warmup_rate, args.max_ans_length))
    args = utils.check_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    # load the bert setting
    bert_config_file = args.bert_model + "/bert_config.json"
    bert_config = BertConfig.from_json_file(bert_config_file)
    

    # load data
    print('loading data...')
    vocab_file = args.bert_model + "/vocab.txt"
    tokenizer = tokenization.BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
    assert args.vocab_size == len(tokenizer.vocab)
    if not os.path.exists(args.train_dir):
        json2features(args.train_file, [args.train_dir.replace('_features_', '_examples_'), args.train_dir],
                  tokenizer, is_training=True,
                  max_seq_length=bert_config.max_position_embeddings)

    if not os.path.exists(args.dev_dir1) or not os.path.exists(args.dev_dir2):
        json2features(args.dev_file, [args.dev_dir1, args.dev_dir2], tokenizer, is_training=False,
                  max_seq_length=bert_config.max_position_embeddings)

    train_features = json.load(open(args.train_dir, 'r'))
    dev_examples = json.load(open(args.dev_dir1, 'r'))
    dev_features = json.load(open(args.dev_dir2, 'r'))
    log_file = args.output_dir+"/log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    steps_per_epoch = len(train_features) // args.n_batch
    eval_steps = int(steps_per_epoch * args.eval_epochs)
    dev_steps_per_epoch = len(dev_features) // args.n_batch
    if len(train_features) % args.n_batch != 0:
        steps_per_epoch += 1
    if len(dev_features) % args.n_batch != 0:
        dev_steps_per_epoch += 1
    total_steps = steps_per_epoch * args.train_epochs

    print('steps per epoch:', steps_per_epoch)
    print('total steps:', total_steps)
    print('warmup steps:', int(args.warmup_rate * total_steps))

    seed_ = args.seed
    with open(log_file, 'a') as aw:
        aw.write('===================================' +
                 'SEED:' + str(seed_)
                 + '===================================' + '\n')
    print('SEED:', seed_)

    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_)

    # init model
    print('init model...')
    model = BertForQuestionAnswering(bert_config)
    
    utils.torch_show_all_params(model)
    init_restore_dir = args.bert_model + "/pytorch_model.bin"
    utils.torch_init_model(model, init_restore_dir)
    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    optimizer = get_optimization(model=model,
                                 float16=args.float16,
                                 learning_rate=args.lr,
                                 total_steps=total_steps,
                                 schedule=args.schedule,
                                 warmup_rate=args.warmup_rate,
                                 max_grad_norm=args.clip_norm,
                                 weight_decay_rate=args.weight_decay_rate)

    all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features], dtype=torch.long)

    seq_len = all_input_ids.shape[1]
    # 样本长度不能超过bert的长度限制
    assert seq_len <= bert_config.max_position_embeddings

    # true label
    all_start_positions = torch.tensor([f['start_position'] for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f['end_position'] for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_start_positions, all_end_positions)
    train_dataloader = DataLoader(train_data, batch_size=args.n_batch, shuffle=True)

    print('***** Training *****')
    model.train()
    global_steps = 1
    best_f1_em = 0
    best_em = 0
    best_f1 = 0
    for i in range(int(args.train_epochs)):
        print('Starting epoch %d' % (i + 1))
        total_loss = 0
        iteration = 1
        with tqdm(total=steps_per_epoch, desc='Epoch %d' % (i + 1)) as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                total_loss += loss.item()
                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (iteration + 1e-5))})
                pbar.update(1)

                if args.float16:
                    optimizer.backward(loss)
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used and handles this automatically
                    lr_this_step = args.lr * warmup_linear(global_steps / total_steps, args.warmup_rate)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                else:
                    loss.backward()

                optimizer.step()
                model.zero_grad()
                global_steps += 1
                iteration += 1

                #if global_steps % eval_steps == 0:
            best_f1, best_em, best_f1_em = evaluate(model, args, dev_examples, dev_features, device,
                                                        global_steps, best_f1, best_em, best_f1_em)
    

            print('Best F1:', best_f1, 'Best EM:', best_em)
            with open(log_file, 'a') as aw:
                aw.write('Best F1:{}\n'.format(best_f1))
                aw.write('Best EM:{}\n'.format(best_em))
        
    # release the memory
    del model
    del optimizer
    torch.cuda.empty_cache()
