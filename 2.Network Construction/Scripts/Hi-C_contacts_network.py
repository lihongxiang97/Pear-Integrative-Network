#! /usr/bin/env python
#Writed by LiHongxiang on 3/9/2023
#本脚本用来从fit-hic的输出的loop中，提取基因和基因之间的连结，用来构建网络
import argparse
import pandas as pd
parser = argparse.ArgumentParser(description="Get gene-gene connects from outfile of fithic2 to construct network!")
parser.add_argument("-G","--gff",required=True,help="gff,get by 'prepare_gff-v2.pl'")
parser.add_argument("-L","--loop",required=True,help="interaction loop,output by fithic2,format:Chr1  fragmentMid1    chr2 fragmentMid2 contacts")
parser.add_argument("-r",required=True,help="resolution/bin of loops")
parser.add_argument("-o",required=True,help="name of output file")
args = parser.parse_args()

gff_file = args.gff
loop_file = args.loop
resolution = args.r
out = args.o

def read_f(file):
    f = open(file).readlines()
    return f

def get_connect(gff,loop):
    gene1 = []
    gene2 = []
    contacts = []
    d = {}
    for looplines in loop:
        loopline = looplines.strip().split()
        a=0
        b=0

        loop_gene1=[]
        loop_gene2=[]

        for gfflines in gff:
            gffline = gfflines.strip().split()
            if '-' in gffline[0]:
                chr = gffline[0].split('-')[1]
            else:
                chr = gffline[0]
            if loopline[0] == chr:
                if int(loopline[1])-int(resolution)/2 <= int(gffline[3]) and int(loopline[1])+int(resolution)/2 >= int(gffline[2]):
                    a = 1
                    loop_gene1.append(gffline[1])
            if loopline[2] == chr:
                if int(loopline[3])-int(resolution)/2 <= int(gffline[3]) and int(loopline[3])+int(resolution)/2 >= int(gffline[2]):
                    b = 1
                    loop_gene2.append(gffline[1])

        if a == 1 and b ==1:
            for j in loop_gene1:
                for k in loop_gene2:
                    if j == k:
                        pass
                    else:
                        if j+','+k not in d and k+','+j not in d:
                            d[j+','+k] = int(loopline[4])
                        elif j+','+k in d:
                            d[j+','+k] += int(loopline[4])
                        elif k+','+j in d:
                            d[k+','+j] += int(loopline[4])

    for key,value in d.items():
        a, b = key.split(',')
        gene1.append(a)
        gene2.append(b)
        contacts.append(value)

    dic = {"source":gene1,
           "target":gene2,
           "contacts":contacts}
    data = pd.DataFrame(dic)

    return data

gff = read_f(gff_file)
loop = read_f(loop_file)
df = get_connect(gff=gff,loop=loop)

df.iloc[:,2] = pd.to_numeric(df.iloc[:,2])
df.iloc[:,2] = df.iloc[:,2]/(2*max(df.iloc[:,2]))+0.5
df.columns = df.columns.str.replace('contacts','weight')
df.to_csv(out,index=False)
