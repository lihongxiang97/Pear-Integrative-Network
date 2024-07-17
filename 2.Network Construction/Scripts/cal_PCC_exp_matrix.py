import argparse
import pandas as pd
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description="Calculate PCC of gene expression matrix!")
parser.add_argument("-m", required=True, help="Gene expression matrix, format is csv, the first row is sample id, the first col is gene id.")
parser.add_argument("-p", required=True, help="gene pairs")
parser.add_argument("-o", required=True, help="output file: PCC file, format is four col:gene_id1\tgene_id2\tpearson_correlation\tp_value")
args = parser.parse_args()
def compute_pearson(gene_pair, gene_expression):
    """
    计算两个基因之间的皮尔森相关系数和p值
    """
    gene1, gene2 = gene_pair
    gene1_expression = gene_expression.loc[gene1]
    gene2_expression = gene_expression.loc[gene2]
    pearson_correlation, p_value = pearsonr(gene1_expression, gene2_expression)
    return pearson_correlation, p_value

# 读取基因表达矩阵csv文件
expression_df = pd.read_csv(args.m, sep='\t', index_col=0)
out = open(args.o,'w')
with open(args.p) as f:
    for lines in f:
        line = lines.strip().split()
        gene_pair = [line[0],line[1]]
        pearson_correlation, p_value = compute_pearson(gene_pair,expression_df)
        print(line[0],line[1],pearson_correlation, p_value,sep='\t',file=out)
out.close()
