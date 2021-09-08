# -*- coding: utf-8 -*-
import os
import sys
import codecs
import json
import base64
import re
import tokenization
from pyltp import SentenceSplitter

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)
    
vocab_file = "ernie_base/vocab.txt"
do_lower_case = True

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
sep_id = tokenizer.convert_tokens_to_ids(["sepsepsep"])[0]

fout_text = open('/data1/bettyleihe/topbase_docs/html_linked_filtered/raw_CH/token_09','w')
fout_ent = open('/data1/bettyleihe/topbase_docs/html_linked_filtered/raw_CH/entity_09','w')

for line in open('/data1/bettyleihe/topbase_docs/html_linked_filtered/ann_CH/ann_09','r'):
    doc = line.strip()
    segs = doc.split("[__end__]")
    content = segs[0]  ## utf8 encoded
    sentences = SentenceSplitter.split(content)
    #print "|||@@|||".join(sentences)
    map_segs = segs[1:]
    maps = {}
    for x in map_segs:
        v = x.split("[__map__]")
        if len(v) != 2:
            continue
        maps[v[0]] = v[1]
        
    text_out = [len(sentences)]
    ent_out = [len(sentences)]

    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        anchor_segs = [x.strip() for x in sent.split("sepsepsep")]
        result = []
        for x in anchor_segs:
            if x in maps:
                result.append(maps[x])
            else:
                result.append("#UNK#")
        cur_seg = 0

        new_text_out = []
        new_ent_out = []

        for token in tokenizer.convert_tokens_to_ids(tokens):
            if token != sep_id:
                new_text_out.append(token)
                new_ent_out.append(result[cur_seg])
            else:
                cur_seg += 1
        
        if len(new_ent_out) != 0:
            ent_out.append(len(new_ent_out))
            ent_out.extend(new_ent_out)
            text_out.append(len(new_text_out))
            text_out.extend(new_text_out)
        else:
            text_out[0] -= 1
            ent_out[0] -= 1
    fout_ent.write("\t".join([str(x) for x in ent_out])+"\n")
    fout_text.write("\t".join([str(x) for x in text_out])+"\n")
    
fout_ent.close()
fout_text.close()
    
        
        
