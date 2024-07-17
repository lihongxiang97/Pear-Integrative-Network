import sys
import pandas as pd
if len(sys.argv) != 3:
    print('Usage:python get_hub_genes.py nodeinfo.txt hub_gene_file')
else:
    df = pd.read_csv(sys.argv[1],sep='\t')
    degree_90th_percentile = df['degree'].quantile(0.90)
    hub_gene_rows = df[df['degree'] > degree_90th_percentile]
    hub_gene = hub_gene_rows['Gene']
    hub_gene.to_csv(sys.argv[2],index=False)