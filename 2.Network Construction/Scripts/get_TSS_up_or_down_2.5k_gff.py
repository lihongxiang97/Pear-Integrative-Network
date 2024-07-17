#! /usr/bin/env python
#Writed by LiHongxiang on 3/16/2023
#本脚本通过gff3和基因组fasta文件，提取基因转录起始位点上下游2.5kb的序列
import argparse
from Bio import SeqIO
parser = argparse.ArgumentParser(description="Extract promoter sequences from GFF3 and FASTA files!")
parser.add_argument("-G","--gff",required=True,help="path to GFF3 file")
parser.add_argument("-A","--fasta",required=True,help="path to genome FASTA file")
parser.add_argument("-up",required=True,help="name of output up file")
parser.add_argument("-down",required=True,help="name of output down file")
parser.add_argument('-upstream', type=int, default=2500, help='number of base pairs upstream of TSS to extract')
parser.add_argument('-downstream', type=int, default=2500, help='number of base pairs downstream of TSS to extract')
args = parser.parse_args()

#反向互补函数
def reverse_complement(seq):
    d={'A':'T','T':'A','C':'G','G':'C','a':'T','t':'A','c':'G','g':'C','N':'N','n':'N'}
    s=''
    for i in seq[::-1]:
        s+=d[i]
    return s

# 读取基因组fasta文件
genome_file = args.fasta
genome = SeqIO.index(genome_file, "fasta")

# 读取gff3文件，提取转录起始位点和基因ID信息
gff_file = args.gff
tss_dict = {}
for line in open(gff_file):
    if not line.startswith("#"):
        fields = line.strip().split("\t")
        if fields[2] == "gene":
            gene_id = fields[-1].split(";")[0].replace("ID=", "")
        elif fields[2] == "mRNA":
            if fields[6] == "+":
                tss = [int(fields[3]),"+",fields[0]]
            if fields[6] == "-":
                tss = [int(fields[4]),"-",fields[0]]
            tss_dict[gene_id] = tss

# 提取转录起始位点上游2.5kb序列，并生成fasta文件
up_file = open(args.up, 'w')
up = args.upstream
for gene_id, tss in tss_dict.items():
    chrom = tss[2]
    strand = tss[1]
    if strand == "+":
        start = max(tss[0] - up - 1, 0)
        end = tss[0] - 1
        seq = genome[chrom][start:end].seq
    else:
        start = tss[0]
        end = min(tss[0] + up, len(genome[chrom]))
        seq_i = genome[chrom][start:end].seq
        seq = reverse_complement(seq_i)
    print('>'+gene_id,file=up_file)
    print(seq,file=up_file)

# 提取转录起始位点下游2.5kb序列，并生成fasta文件
down_file = open(args.down, 'w')
down = args.downstream
for gene_id, tss in tss_dict.items():
    chrom = tss[2]
    strand = tss[1]
    if strand == "+":
        start = tss[0] #genome索引从0开始，所以tss对应的就是tss的下一位
        end = min(tss[0] + down, len(genome[chrom]))
        seq = genome[chrom][start:end].seq
    else:
        start = max(tss[0] - down - 1, 0)
        end = tss[0] - 1
        seq_i = genome[chrom][start:end].seq
        seq = reverse_complement(seq_i)
    print('>'+gene_id,file=down_file)
    print(seq,file=down_file)
