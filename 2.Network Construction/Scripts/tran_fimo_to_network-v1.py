#! /usr/bin/env python
#Writed by LiHongxiang on 10/31/2023
#本脚本用来将v1版fimo结果计算weight，得到network文件
import argparse
import math
import pandas as pd
parser = argparse.ArgumentParser(description='Replace the v1 gene by v2 and calculate the weight of edge')
parser.add_argument('-f', help='fimo file')
parser.add_argument('-t', help='"up" or "down" to upstream or downstream of tss faste')
parser.add_argument('-o', help='output network file')
args = parser.parse_args()

source = []
target = []
weight = []
merge_score = {}

with open(args.f, 'r') as f1:
    for lines in f1.readlines()[1:]:
        if not lines.startswith('#'):
            if lines != '\n':
                line = lines.strip().split()
                if args.t == "up":
                    distance = 2501 - (int(line[3])+int(line[4]))/2
                if args.t == "down":
                    distance = (int(line[3]) + int(line[4])) / 2
                score = float(line[6])*math.exp(-distance/2500)
                if line[0]+'\t'+line[2] not in merge_score:
                    merge_score[line[0]+'\t'+line[2]] = score
                else:
                    merge_score[line[0]+'\t'+line[2]] += score
for key,value in merge_score.items():
    gene1, gene2 = key.split('\t')
    source.append(gene1)
    target.append(gene2)
    weight.append(value)

dic = {"source":source,
        "target":target,
        "weight":weight}
df = pd.DataFrame(dic)
#df.iloc[:,2] = pd.to_numeric(df.iloc[:,2])
#df.iloc[:,2] = df.iloc[:,2]/(2*max(df.iloc[:,2]))+0.5
df.to_csv(args.o,index=False)