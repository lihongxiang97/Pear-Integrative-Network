#! /usr/bin/env python
#Writed by LiHongxiang on 3/20/2023
#本脚本用来将tss上游序列和下游序列结合的转录因子网络合并，score取和，并计算weight
import argparse
import pandas as pd
parser = argparse.ArgumentParser(description="Merge tss up and down TF PWM networks!")
parser.add_argument("-up",required=True,help="network file by format:node1,node2,weight")
parser.add_argument("-down",required=True,help="network file by format:node1,node2,weight")
parser.add_argument("-o",required=True,help="name of output file.csv")
args = parser.parse_args()

d = {}
with open(args.up) as f:
    for lines in f.readlines()[1:]:
        line = lines.strip().split(',')
        d[line[0]+','+line[1]] = float(line[2])

with open(args.down) as f:
    for lines in f.readlines()[1:]:
        line = lines.strip().split(',')
        if line[0]+','+line[1] in d:
            d[line[0] + ',' + line[1]] += float(line[2])
        else:
            d[line[0] + ',' + line[1]] = float(line[2])
source = []
target = []
weight = []
for key,value in d.items():
    a, b = key.split(',')
    source.append(a)
    target.append(b)
    weight.append(value)

dic = {'source':source,
       'target':target,
       'weight':weight}
df = pd.DataFrame(dic)
df.iloc[:,2] = pd.to_numeric(df.iloc[:,2])
df.iloc[:,2] = df.iloc[:,2]/(2*max(df.iloc[:,2]))+0.5
df.to_csv(args.o,index=False)


