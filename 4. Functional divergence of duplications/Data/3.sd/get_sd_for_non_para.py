import sys
import pandas as pd
import random
if len(sys.argv) != 4:
    print('Usage: python get_sd.py Non-paralogous.genes sd_file type')
else:
    gene = []
    gene_pair = []
    with open(sys.argv[1]) as f:
        for lines in f:
            line = lines.strip()
            gene.append(line)
    for _ in range(10000):
        select_gene_pairs = random.sample(gene,2)
        gene_pair.append(select_gene_pairs)

    sd_matrix = pd.read_parquet(sys.argv[2])
    for i in gene_pair:
        if i[0] in sd_matrix.index and i[1] in sd_matrix.index:
            sd = sd_matrix[i[0]][i[1]]
            print(sys.argv[3],sd,sep='\t')
