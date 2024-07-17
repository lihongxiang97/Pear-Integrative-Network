#! /usr/bin/env python
#Writed by LiHongxiang on 6/27/2023
import argparse
import math
parser = argparse.ArgumentParser(description='Replace the v1 gene by v2 and calculate the weight of edge')
parser.add_argument('-f', help='fimo file')
parser.add_argument('-p', help='peak name')
parser.add_argument('-o', help='output network file')
args = parser.parse_args()

peak_gene_contact = {}
with open(args.p) as peak_gene_file:
    for lines in peak_gene_file:
        line = lines.strip().split()
        if line[0] not in peak_gene_contact:
            peak_gene_contact[line[0]] = [(line[1],line[2])]
        else:
            peak_gene_contact[line[0]].append((line[1],line[2]))

with open(args.f, 'r') as fimo_file, open(args.o,'w') as output:
    for lines in fimo_file.readlines()[1:]:
        if not lines.startswith('#'):
            if lines != '\n':
                line = lines.strip().split()
                TF_id = line[0]
                for i in peak_gene_contact[line[2]]:
                    gene_id, contact = i
                    print(TF_id,gene_id,contact,sep='\t',file=output)
