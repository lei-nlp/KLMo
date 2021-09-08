import random
import sys
import numpy as np
import collections
import torch
import tensorflow as tf
import codecs
import indexed_dataset
import json

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)


flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_file_prefix", None,
                    "Input text/entity file.")
flags.DEFINE_string(
        "output_file", None,
        "Output TF example file (or comma-separated list of files).")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
flags.DEFINE_integer("max_predictions_per_seq", 40,
                     "Maximum number of masked LM predictions per sequence.")
flags.DEFINE_integer("max_entity_per_seq", 20,
                      "Maximum number of entities per sequence.")
flags.DEFINE_integer("max_relation_per_seq", 30,
                      "Maximum number of relations per sequence.")
flags.DEFINE_integer("random_seed", 123, "Random seed for data generation.")
flags.DEFINE_integer(
        "dupe_factor", 3,
        "Number of times to duplicate the input data (with different masks).")
flags.DEFINE_float("masked_lm_prob", 0.05, "Masked LM probability.")
flags.DEFINE_float("masked_ent_prob", 0.5, "Masked ENT probability.")
flags.DEFINE_float(
        "short_seq_prob", 0.1,
        "Probability of creating sequences which are shorter than the "
        "maximum length.")
        
vocab_words_size = 21127

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

# read uuid2type mapping
with open("/apdcephfs/common/bettyleihe/Pretrained_Encyclopedia/fusion_info/uuid_2_type_dict", 'r') as fin:
    uuid2type = {}
    #fin.readline()
    while 1:
        l = fin.readline()
        if l == "":
            break
        parts = l.strip().split('\t')
        if len(parts) != 2:
            print (l.encode('utf8'))
            continue
        uuid2type[parts[0].strip()] = int(parts[1].strip())
'''
# read uuid_2_entity_name mapping
with codecs.open("/apdcephfs/common/bettyleihe/Pretrained_Encyclopedia/fusion_info/uuid_2_entity_name", 'r','utf8') as fin:
    uuid_2_entity_name = {}
    while 1:
        l = fin.readline()
        if l == "":
            break
        parts = l.strip().split('\t')
        if len(parts) != 2:
            print(l.encode('utf8'))
            continue
        uuid_2_entity_name[parts[0].strip()] = parts[1].strip()

# read type_2_uuid_list mapping
with open("/apdcephfs/common/bettyleihe/Pretrained_Encyclopedia/fusion_info/type_2_uuid_list", 'r') as fin:
    type_2_uuid_list = {}
    #fin.readline()
    while 1:
        l = fin.readline()
        if l == "":
            break
        parts = l.strip().split()
        tid = parts[0].strip()
        uuids = parts[1:]
        uuids = [x.strip() for x in uuids]
        type_2_uuid_list[int(tid)] = uuids
'''
'''
# read type_2_cand_list mapping
type_2_cands = {}
for line in open("/apdcephfs/common/bettyleihe/Pretrained_Encyclopedia/fusion_info/type_2_cand_dict", 'r', encoding='UTF-8'):
    line = line.strip()
    data_dict = json.loads(line)
    type_id = data_dict["type"]
    cand_list = data_dict["cand"]
    if len(cand_list) < 500:
        continue
    type_2_cands[type_id] = cand_list
print (len(type_2_cands)) 
'''
'''
# read type_2_cand_list mapping
type_2_cands = {}
for line in open("/apdcephfs/common/bettyleihe/Pretrained_Encyclopedia/fusion_info/type_2_cand_dict.small.withlen", 'r', encoding='UTF-8'):
    line = line.strip()
    type_2_cands_ = json.loads(line)
for k, v_ in type_2_cands_.items():
    v = dict()
    for vk, vv in v_.items():
        v[int(vk)] = vv
    type_2_cands[int(k)] = v
print (len(type_2_cands)) 
'''
# read entity mapping
uuid_2_id = {}
with open("/apdcephfs/common/bettyleihe/Pretrained_Encyclopedia/kg_embed_CH/entity2id.txt", 'r') as fin:
    fin.readline()
    while 1:
        l = fin.readline()
        if l == "":
            break
        ent, idx = l.strip().split()
        uuid_2_id[ent] = int(idx)
print (len(uuid_2_id))

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
    
class TrainingInstance(object):
    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
            is_random_next):
        self.input_ids = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

def create_training_instances(input_file, output_file, max_seq_length,
        dupe_factor, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, max_entity_per_seq, max_relation_per_seq, rng):

    all_documents = []
    all_documents_ent = []
    #all_doc_eids = []
    with tf.gfile.GFile(input_file+"_token", "r") as reader:
        with tf.gfile.GFile(input_file+"_entity", "r") as reader_ent:
            while True:
                line = reader.readline()
                line_ent = reader_ent.readline()
                # if len(all_documents) > 10:
                #     break
                if not line:
                    break
                line = [int(x) for x in line.strip().split()]
                vec = line_ent.strip().split()
                #eid_list = []
                for i, x in enumerate(vec):
                    if x == "#UNK#":
                        vec[i] = '-1'
                    #elif x[0] == "Q":
                    else:
                        #if x in uuid2type and uuid2type[x] != 61 and uuid2type[x] != 230 and uuid2type[x] in type_2_cands:
                        if x in uuid2type and uuid2type[x] != 61 and uuid2type[x] != 230 and x in uuid_2_id: # and uuid2type[x] in type_2_cands:
                            vec[i] = x
                            #if uuid_2_id[x] not in eid_list:
                            #    eid_list.append(uuid_2_id[x])
                            #if i != 0 and vec[i] == vec[i-1]:
                            #    vec[i] = -1 # Q123 Q123 Q123 -> d[Q123] -1 -1
                        else:
                            vec[i] = '-1'
                    #else:
                    #    vec[i] = int(x)
                if line[0] != 0:
                    all_documents.append(line)
                    all_documents_ent.append(vec)
                    #all_doc_eids.append(eid_list)
    #seed = rng.randint(0,100)
    #rng.seed(seed)
    #rng.shuffle(all_documents)
    #rng.seed(seed)
    #rng.shuffle(all_documents_ent)
    for epoch_ in range(dupe_factor):
        seed = rng.randint(0,100)
        rng.seed(seed)
        rng.shuffle(all_documents)
        rng.seed(seed)
        rng.shuffle(all_documents_ent)
        #rng.seed(seed)
        #rng.shuffle(all_doc_eids)
        ds = indexed_dataset.IndexedDatasetBuilder(output_file+"_"+str(epoch_+1)+".bin")
        for document_index in range(len(all_documents)):
            create_instances_from_document(
                ds, all_documents, all_documents_ent, document_index, max_seq_length, short_seq_prob,
                masked_lm_prob, max_predictions_per_seq, max_entity_per_seq, max_relation_per_seq, rng, epoch_)
        ds.finalize(output_file+"_"+str(epoch_+1)+".idx")

def jump_in_document(document, i):
    pos = 1
    while i > 0:
        pos = pos + 1 + document[pos]
        i -= 1
    return pos

def create_instances_from_document(
        ds, all_documents, all_documents_ent, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, max_entity_per_seq, max_relation_per_seq, rng, epoch_):
    document = all_documents[document_index]
    document_ent = all_documents_ent[document_index]
    
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 2
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)
    current_chunk = []
    current_length = 0
    i = 0
    while i < document[0]:
        current_chunk.append(i)
        current_length += document[jump_in_document(document, i)]
        if i == document[0] - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                #a_end = 1
                #if len(current_chunk) >= 2:
                #    a_end = rng.randint(1, len(current_chunk) - 1)
                tokens_a = []
                entity_a = []
                #for j in current_chunk[:a_end]:
                for j in current_chunk:
                    pos = jump_in_document(document, j)
                    tokens_a.extend(document[pos+1:pos+1+document[pos]])
                    entity_a.extend(document_ent[pos+1:pos+1+document[pos]])
                
                truncate_seq_pair(tokens_a, entity_a, max_num_tokens)

                assert len(tokens_a) >= 1
                #assert len(tokens_b) >= 1

                tokens = [101] + tokens_a + [102] #+ tokens_b + [102]
                entity = ['-1'] + entity_a + ['-1'] #+ entity_b + ['-1']
                
                assert len(tokens) == len(entity)
                segment_ids = [0]*(len(tokens_a)+2) #+ [1]*(len(tokens_b)+1)
                
                
                (ent_before_positions, ent_after_positions, ent_ids, ent_uniq_ids) = create_ent_predictions(tokens, entity)
                er_ids = gen_ER_list(ent_uniq_ids)  ## generate entity+relation input: r1 e1 e2 r2 e1 e3 ...
                
                ent_times = len(ent_before_positions)
 
                (tokens, masked_lm_positions,
                 masked_lm_ids) = create_masked_lm_predictions(
                         tokens, entity, masked_lm_prob, max_predictions_per_seq, rng)

                input_ids = tokens
                input_mask = [1] * len(input_ids)
                assert len(input_ids) <= max_seq_length
                if len(input_ids) < max_seq_length:
                    rest = max_seq_length - len(input_ids)
                    input_ids.extend([0]*rest)
                    input_mask.extend([0]*rest)
                    segment_ids.extend([0]*rest)
                
                masked_lm_labels = np.ones(len(input_ids), dtype=int)*-1
                masked_lm_labels[masked_lm_positions] = masked_lm_ids
                masked_lm_labels = list(masked_lm_labels)
               
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
                ##er_segments: 111 000 111
                #for c in range(len(er_segments)):
                #    if (c/3) % 2 == 0:
                #        er_segments[c] = 1
                #    else:
                #        er_segments[c] = 0
                if len(er_ids) >= max_er_len:
                    er_ids = er_ids[:max_er_len]
                    er_mask = er_mask[:max_er_len]
                    er_segments = er_segments[:max_er_len]
                else:
                    rest = max_er_len - len(er_ids)
                    er_ids.extend([-1]*rest)
                    er_mask.extend([0]*rest)
                    er_segments.extend([1]*rest)
                
                #next_sentence_label = 1 if is_random_next else 0

                if ent_times >= 2:
                    ds.add_item(torch.IntTensor(input_ids+input_mask+segment_ids+masked_lm_labels+ent_before_positions+ent_after_positions+ent_ids+er_ids+er_mask+er_segments))


            current_chunk = []
            current_length = 0
        i+=1
        
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

def create_masked_lm_predictions(tokens, entity, masked_lm_prob,
        max_predictions_per_seq, rng):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == 101 or token == 102 or entity[i] != '-1':
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq,
            max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = None
        if rng.random() < 0.8:
            masked_token = 103 # [MASK]
        else:
            if rng.random() < 0.5:
                masked_token = tokens[index]
            else:
                 masked_token = rng.randint(0, vocab_words_size - 1)
        output_tokens[index] = masked_token
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)
    
    
def create_ent_predictions(tokens, entity):
    ent_left_indexes = []
    ent_right_indexes = []
    ent_ids = []
    ent_uniq_ids = []
    for (i, ent) in enumerate(entity):
        if i > 0 and entity[i] != '-1' and entity[i-1] == '-1' and entity[i] in uuid_2_id:
            ent_left_indexes.append(i-1)
            eid = uuid_2_id[entity[i]]
            ent_ids.append(eid)
            if eid not in ent_uniq_ids:
                ent_uniq_ids.append(eid)
        if i < len(entity)-1 and entity[i] != '-1' and entity[i+1] == '-1' and entity[i] in uuid_2_id:
            ent_right_indexes.append(i+1)
    assert len(ent_left_indexes) == len(ent_right_indexes)
    assert len(ent_ids) == len(ent_right_indexes)
    return (ent_left_indexes, ent_right_indexes, ent_ids, ent_uniq_ids)

                 
                 
def truncate_seq_pair(tokens_a, entity_a, max_num_tokens):
    while True:
        total_length = len(tokens_a)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a 
        trunc_entity = entity_a 
        assert len(trunc_tokens) >= 1
        trunc_tokens.pop()
        trunc_entity.pop()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("*** Reading from input files ***")
    tf.logging.info("%s", FLAGS.input_file_prefix)

    rng = random.Random(FLAGS.random_seed)

    create_training_instances(
            FLAGS.input_file_prefix, FLAGS.output_file, FLAGS.max_seq_length, FLAGS.dupe_factor,
            FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq, FLAGS.max_entity_per_seq, FLAGS.max_relation_per_seq,
            rng)

    #tf.logging.info("*** Writing to output files ***")
    #tf.logging.info("%s", FLAGS.output_file)
    #write_instance_to_example_files(instances, FLAGS.max_seq_length,
    #        FLAGS.max_predictions_per_seq, FLAGS.output_file, FLAGS.vocab_file)

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file_prefix")
    flags.mark_flag_as_required("output_file")
    tf.app.run()

