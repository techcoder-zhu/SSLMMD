import pipeline
import train
import pandas as pd
import os

src_project_list = ['camel', 'poi',  'xalan', 'xerces'] # plz add more source projects
dst_project_list = ['ant', 'camel', 'jedit', 'log4j', 'lucene', 'poi', 'synapse', 'velocity', 'xalan', 'xerces']  # plz add more target projects including all source projects
# src_project_list = ['xalan'] # plz add more source projects
# dst_project_list = ['xalan', 'ant']  # plz add more target projects including all source projects
ppl = pipeline.Pipeline('4:1', 'data/')
print('parse source code...')
ppl.parse_source('dataset/')
for pro in dst_project_list:
    print('merge versions of ' + pro + '...')
    ppl.merge_versions(pro)

for src_pro in src_project_list:
    dst = dst_project_list.copy()
    dst.remove(src_pro)
    for dst_pro in dst:
        print('split data...')
        ppl.split_data(src_pro, dst_pro)
        print('train word embedding...')
        ppl.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        ppl.generate_block_seqs(ppl.train_file_path, 'train')
        ppl.generate_block_seqs(ppl.dev_file_path, 'dev')
        ppl.generate_block_seqs(ppl.test_file_path, 'test')
        print('training on ' + src_pro + ' and test on '+ dst_pro +'...')
        train.run(src_pro,dst_pro)
