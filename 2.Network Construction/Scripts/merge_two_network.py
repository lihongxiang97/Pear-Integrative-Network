#! /usr/bin/env python
#Writed by LiHongxiang on 3/10/2023
#本脚本用来将两个网络合并，weight值取二者之和
import argparse
parser = argparse.ArgumentParser(description="Merge two networks!")
parser.add_argument("-n1",required=True,help="network file by format:node1\tnode2\tweight")
parser.add_argument("-n2",required=True,help="network file by format:node1\tnode2\tweight")
parser.add_argument("-o",required=True,help="name of output file.csv")
args = parser.parse_args()

f1 = open(args.n1).readlines()
f2 = open(args.n2).readlines()
edges = {}

for lines in f1[1:]:
    if ',' in lines:
        node1, node2, weight = lines.strip().split(',')[0:3]
    else:
        node1, node2, weight = lines.strip().split('\t')[0:3]
    key = (node1,node2)
    edges[key] = abs(float(weight))

for lines in f2[1:]:
    if ',' in lines:
        node1, node2, weight = lines.strip().split(',')[0:3]
    else:
        node1, node2, weight = lines.strip().split('\t')[0:3]
    key = (node1, node2)
    if key in edges:
        edges[key] += abs(float(weight))
    elif (node2,node1) in edges:
        edges[(node2, node1)] += abs(float(weight))
    else:
        edges[key] = abs(float(weight))

# Write the merged edges to the output file
with open(args.o, "w") as f:
    f.write(f"source,target,weight\n")
    for key, value in edges.items():
        node1, node2 = key
        f.write(f"{node1},{node2},{value}\n")






