# KLMo
知识图谱增强的中文预训练模型 KLMo: Knowledge Graph Enhanced Pretrained Language Model with Fine-Grained Relationships

test
1. 开发环境要求：
  pytorch 1.3.1
  tensorflow 1.14
  Python 3
  tqdm
  boto3
  requests
  apex
  升级gcc版本到7.3
  
2. 预训练数据准备
   预训练数据是Topbase知识库中的百度百科网页，将超链接的锚文本对应到知识库中实体的uuid，并补充实体间的关系id。
   关系的补充方法：获取Topbase知识库三元组，对文本中的任意两个实体，在KB三元组集合中检索，若存在包含这两个实体的三元组，则将该三元组作为输入补充进来。
   
   2.1 Topbase百度百科网页数据
   地址：root@100.115.135.106:/data/bettyleihe/topbase_docs/html_linked_filtered/splits
   格式：实体ID \t uuid \t url \t 网页文本
   样例：7137710550718529747	4316560c-e1f3-4887-ab5d-64c5717346d3	http://baike.baidu.com/subview/160147/9785409.htm	《再说一次我爱你》是香港艺人{{刘德华||ad99c1e1-1209-46d5-b63c-f3ddad959479,视频类_影视明星}}演唱的一首歌曲，由刘德华、{{李安修||460fe650-b717-4b00-8265-c4a396261e61,音乐类_歌手}}作词、Kim Hom Jick作曲。收录在刘德华2005年8月发行的同名专辑《{{再说一次我爱你||7e0b082b-b576-4793-8334-1423ef339dd8,音乐类_专辑}}》中。
   
   2.2 百科数据预处理脚本
   1） pretrain_data/ernie_proprecess.py：原始数据生成ann_CH目录中的格式化文本，地址：root@100.102.32.61:/data1/bettyleihe/topbase_docs/html_linked_filtered/ann_CH
   2） pretrain_data/create_ids.py: 将ann_CH目录中的文本转化为raw_CH中的实体、token对齐的文本，地址：root@100.102.32.61:/data1/bettyleihe/topbase_docs/html_linked_filtered/raw_CH
   3） pretrain_data/create_insts.py：调用code/create_instances_addRelation.py生成训练实例,bin文件，地址：root@9.21.146.66:/data1/bettyleihe/apdcephfs_common.bak/Pretrained_Encyclopedia/pretrain_data/data_addRel_all
   4） code/merge_all.py: 训练数据合并，地址：root@9.21.146.66:/data1/bettyleihe/apdcephfs_common.bak/Pretrained_Encyclopedia/pretrain_data/merge_addRel_all.bin 和 merge_addRel_all.idx
   
3. 预训练命令：
   nohup python3 code/run_pretrain_addRelation_T2.py --do_train --data_dir pretrain_data/merge_addRel_all --bert_model bert_base_chinese --output_dir KLMo_out_addRel_T2 --task_name pretrain  --fp16 --max_seq_length 512 --train_batch_size 128 --num_train_epochs 5.0 &
   配置目录：
   1） bert起始模型地址：root@9.21.146.66:/data1/bettyleihe/machine_74.bak/Pretrained_Encyclopedia/bert_base_chinese/pytorch_model.bin
   2） KG图向量地址：root@9.21.146.66:/data1/bettyleihe/machine_74.bak/Pretrained_Encyclopedia/kg_embed_CH/entity2vec.vec.d100 和 relation2vec.vec.d100
 
4. 预训练模型：
   KLMo 预训练模型地址：root@9.21.146.66:/data1/bettyleihe/machine_74.bak/Pretrained_Encyclopedia/KLMo_out/pytorch_model.bin
   
5. 下游任务评测
    5.1 IP作品类实体分类任务
    训练集：data/IP/new.15k.conll.json  测试集：data/IP/new.8k.conll.json
    训练命令：
    nohup python3 code/run_typing_IP_addRel_T2.py --do_train --do_lower_case --data_dir data/IP --ernie_model KLMo_out_CH_d100 --max_seq_length 256 --train_batch_size 128 --learning_rate 2e-5 --num_train_epochs 10.0 --output_dir output_ip_KLMo --threshold 0.0 > res.ip.KLMo 2>&1 &
    测试命令：
    nohup python3 code/eval_typing_IP_addRel_T2.py --do_eval --do_lower_case --data_dir data/IP --ernie_model KLMo_out_CH_d100 --max_seq_length 256 --output_dir output_ip_KLMo --threshold 0.0 > eval.ip.KLMo 2>&1 &
    指标结果：
    	        Micro-P Micro-R Micro-F1 Accuracy
    bert_base	 87.52%	 80.19%	 83.70%	 80.19%
    KLMo	     85.19%	 86.22%	 85.70%	 84.50%


    5.2 人物关系分类任务
    训练集地址：data/IPRE/train.json  测试集：data/IPRE/test.json
    训练命令：
    nohup python3 code/run_ipre_addRel_T2.py --do_train --do_lower_case --data_dir data/IPRE --ernie_model KLMo_out_CH_d100 --max_seq_length 256 --train_batch_size 128 --learning_rate 2e-5 --num_train_epochs 10.0 --output_dir output_ipre_KLMo --threshold 0.0 > res.ipre.KLMo 2>&1 &
    测试命令：
    nohup python3 code/eval_ipre_addRel_T2.py --do_eval --do_lower_case --data_dir data/IPRE --ernie_model KLMo_out_CH_d100 --max_seq_length 256 --output_dir output_ipre_KLMo --threshold 0.0 > eval.ipre.KLMo 2>&1 &
    指标结果：
    	          Micro-P  Micro-R  Micro-F1   
    BERT-base	   15.94%	35.12%	 21.93%	    
    官方baseline	 xx	      xx	 20.56%	     
    KLMo    	   20.90%	31.24%	 25.05%   
