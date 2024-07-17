#! /usr/bin/env python
#Writed by LiHongxiang on 6/26/2023
import argparse
import math
parser = argparse.ArgumentParser(description='Replace the v1 gene by v2 and calculate the weight of edge')
parser.add_argument('-f', help='fimo file')
parser.add_argument('-p', help='peak name')
parser.add_argument('-o', help='output network file')
args = parser.parse_args()

peak_gene_distance = {}
with open(args.p) as peak_gene_file:
    for lines in peak_gene_file:
        line = lines.strip().split(',')
        peak_gene_distance[line[0]] = (line[2],line[3],line[1])

with open(args.f, 'r') as fimo_file, open(args.o,'w') as output:
    for lines in fimo_file.readlines()[1:]:
        if not lines.startswith('#'):
            if lines != '\n':
                line = lines.strip().split()
                TF_id = line[0]
                gene_id, distance, FE = peak_gene_distance[line[2]]
                distance = abs(int(distance))
                FE = float(FE)
                score = FE * math.exp(-distance/2500)
                print(TF_id, gene_id, score, sep='\t',file=output)


