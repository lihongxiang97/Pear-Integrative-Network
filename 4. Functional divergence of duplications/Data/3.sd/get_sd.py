import sys
import pandas as pd
if len(sys.argv) != 4:
    print('Usage: python get_sd.py gene_pairs sd_file type')
else:
    gene_pair = []
    with open(sys.argv[1]) as f:
        for lines in f:
            line = lines.strip().split()
            gene_pair.append(line)
    sd_matrix = pd.read_parquet(sys.argv[2])
    for i in gene_pair:
        if i[0] in sd_matrix.index and i[1] in sd_matrix.index:
            sd = sd_matrix[i[0]][i[1]]
            print(sys.argv[3],sd,sep='\t')
