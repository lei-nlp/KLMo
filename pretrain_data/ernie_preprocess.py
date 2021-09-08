# -*- coding: utf-8 -*-
import os
import sys
import codecs
import json
import base64
import re
from pyltp import SentenceSplitter

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)
    
    
anchor_reg = r'\{\{[^\|\{\}]*?\|\|[a-z\d\-]+,[^\{\}\|,]*?\}\}'
regex = re.compile(anchor_reg)
#fw_text = open('/data/bettyleihe/topbase_docs/html_linked_filtered/raw/token','w')
#fw_ent = open('/data/bettyleihe/topbase_docs/html_linked_filtered/raw/entity','w')
fw_ann = open('/data/bettyleihe/topbase_docs/html_linked_filtered/ann_CH/ann_09','w')
for line in open('/data/bettyleihe/topbase_docs/html_linked_filtered/splits/part_09','r'):
    line = line.strip()
    parts = line.split('\t')
    if len(parts) != 4:
        continue
    doc = parts[3].strip()
    anchor_list = regex.findall(doc)
    #print len(anchor_list)
    lookup = []
    for anchor_item in anchor_list:
        entity_name = anchor_item.split('||')[0][2:]
        typename = anchor_item.split(',')[-1][:-2]
        uuid = anchor_item.split('||')[1].split(',')[0]
        new_part = ' sepsepsep '+entity_name+' sepsepsep '
        #print '{{'+entity_name+'||'+uuid+','+typename+'}}'
        doc = doc.replace(anchor_item, new_part)
        lookup.append((entity_name,uuid))
    lookup = "[__end__]".join(["[__map__]".join(x) for x in lookup])
    fw_ann.write(doc+"[__end__]"+lookup+"\n")
#fw_text.close()
#fw_ent.close()
fw_ann.close()
    
        
        
