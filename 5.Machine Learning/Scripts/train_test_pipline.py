#!/storage/lihongxiang/soft/keras/bin/python3
#Writed by LHX on 2023/11/23
import argparse
parser = argparse.ArgumentParser(description="Machine Learning Pipline for Gene Regulation Network Data")
parser.add_argument("--positive", type=str, required=True, help="Positive gene list file")
parser.add_argument("--negative", type=str, required=True, help="Negative gene list file")
parser.add_argument("--test", type=str, required=True, help="test gene list file")
parser.add_argument("--nodeinfo", type=str, required=True, help="Nodeinfo file of network")
parser.add_argument("--sd", type=str, required=True, help="sd.parquet file of network")
parser.add_argument("--TPM", type=str, required=True, help="TPM file of network")
parser.add_argument("--Protein", type=str, required=True, help="Protein file of network")
parser.add_argument("--train_type", choices=['Multi_transcriptomic','Fd_transcriptomic','Proteomic'], nargs='+', required=True,
                    help="The data type you want to train, you can add one or more type to train data.")
args = parser.parse_args()

import subprocess
import pandas as pd
import time
import numpy as np


def time_print(str):
    print("\033[32m%s\033[0m %s" % (time.strftime('[%H:%M:%S]', time.localtime(time.time())), str))

time_print('Now reading nodeinfo......')
nodeinfo = pd.read_csv(args.nodeinfo,sep='\t',index_col=0)
time_print('Now reading sd......')
sd = pd.read_parquet(args.sd)
#sd = pd.read_csv(args.sd,sep='\t',index_col=0)
time_print('Now reading TPM......')
TPM = pd.read_csv(args.TPM,index_col=0)
#去除所有样本都为0的基因
TPM = TPM.loc[~(TPM == 0).all(axis=1)]
time_print('Now reading protein......')
protein = pd.read_csv(args.Protein,index_col=0)
protein = protein.fillna(0)
protein = np.log10(protein+1)

#Positive
positive_gene = pd.read_csv(args.positive,header=None)
#rm dup
positive_gene = positive_gene.drop_duplicates()
positive_gene.columns = ['Gene']
positive_gene.insert(0, 'Label', 1)
positive_gene.index = positive_gene['Gene']
positive_gene_ids = positive_gene.index
#Negative
negative_gene = pd.read_csv(args.negative,header=None)
#rm dup
negative_gene = negative_gene.drop_duplicates()
negative_gene.columns = ['Gene']
negative_gene.insert(0, 'Label', 0)
negative_gene.index = negative_gene['Gene']
negative_gene_ids = negative_gene.index

gene = pd.concat([positive_gene,negative_gene],axis=0)
#merge id
gene_ids = positive_gene_ids.append(negative_gene_ids)

#筛选没有表达和网络中不存在的基因
existing_index_protein = set([idx for idx in gene_ids if idx in protein.index])
existing_index_tpm = set([idx for idx in gene_ids if idx in TPM.index])
existing_index_node = set([idx for idx in gene_ids if idx in nodeinfo.index])
train_existing_index = list(existing_index_protein.intersection(existing_index_tpm,existing_index_node))

train_sd = sd.loc[train_existing_index, train_existing_index]
train_nodeinfo = nodeinfo.loc[train_existing_index]
train_TPM = TPM.loc[train_existing_index]
train_protein = protein.loc[train_existing_index]
train_gene = gene.loc[train_existing_index]

merged = pd.concat([train_gene, train_sd, train_nodeinfo, train_TPM, train_protein], axis=1)
merged.to_csv('train.csv',index=False)

time_print('Train.csv is done!')
#test
test_gene = pd.read_csv(args.test,header=None)
test_gene = test_gene.drop_duplicates()
test_gene.columns = ['Gene']
test_gene.index = test_gene['Gene']
test_gene_ids = test_gene.index
existing_index_protein = set([idx for idx in test_gene_ids if idx in protein.index])
existing_index_tpm = set([idx for idx in test_gene_ids if idx in TPM.index])
existing_index_node = set([idx for idx in test_gene_ids if idx in nodeinfo.index])
test_existing_index = list(existing_index_protein.intersection(existing_index_tpm,existing_index_node))

test_sd = sd.loc[test_existing_index, train_existing_index]
test_nodeinfo = nodeinfo.loc[test_existing_index]
test_TPM = TPM.loc[test_existing_index]
test_protein = protein.loc[test_existing_index]
test_gene = test_gene.loc[test_existing_index]
merged = pd.concat([test_gene, test_sd, test_nodeinfo, test_TPM, test_protein], axis=1)
merged.to_csv('test.csv',index=False)
time_print('Test.csv is done!')

time_print('Now starting training and testing......')
if 'Multi_transcriptomic' in args.train_type and 'Fd_transcriptomic' in args.train_type and 'Proteomic' in args.train_type:
    subprocess.call('Trainning_results_all.py --train train.csv --test test.csv',shell=True)
elif 'Multi_transcriptomic' in args.train_type and 'Fd_transcriptomic' in args.train_type:
    subprocess.call('Trainning_results_Multi_transcriptomic_Fd_transcriptomic.py --train train.csv --test test.csv', shell=True)
elif 'Fd_transcriptomic' in args.train_type and 'Proteomic' in args.train_type:
    subprocess.call('Trainning_results_Fd_transcriptomic_Proteomic.py --train train.csv --test test.csv', shell=True)
elif 'Multi_transcriptomic' in args.train_type and 'Proteomic' in args.train_type:
    subprocess.call('Trainning_results_Multi_transcriptomic_Proteomic.py --train train.csv --test test.csv', shell=True)
elif 'Multi_transcriptomic' in args.train_type:
    subprocess.call('Trainning_results_Multi_transcriptomic.py --train train.csv --test test.csv', shell=True)
elif 'Fd_transcriptomic' in args.train_type:
    subprocess.call('Trainning_results_Fd_transcriptomic.py --train train.csv --test test.csv', shell=True)
elif 'Proteomic' in args.train_type:
    subprocess.call('Trainning_results_Proteomic.py --train train.csv --test test.csv', shell=True)

time_print('All done!!!The best models are saved in model, the predict results are saved in result!')


