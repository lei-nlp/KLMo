import indexed_dataset
import os

#builder = indexed_dataset.IndexedDatasetBuilder('pretrain_data/merge_entGen_small_dyn_all.bin')
builder = indexed_dataset.IndexedDatasetBuilder('pretrain_data/merge_addRel_all.bin')
#for filename in os.listdir("pretrain_data/data_entGen_small_all_5"):
    #print (filename)
    #if filename[-4:] == '.bin':
        #builder.merge_file_("pretrain_data/data_entGen_small_all_5/"+filename[:-4])
builder.merge_file_("pretrain_data/merge_addRel_0")
builder.merge_file_("pretrain_data/merge_addRel_1")
builder.merge_file_("pretrain_data/merge_addRel_2")
#builder.merge_file_("pretrain_data/data_entGen_small_all_5/merge_entGen_small_dynRatio_3")
#builder.merge_file_("pretrain_data/data_entGen_small_all_5/merge_entGen_small_dynRatio_4")
    
builder.finalize("pretrain_data/merge_addRel_all.idx")
