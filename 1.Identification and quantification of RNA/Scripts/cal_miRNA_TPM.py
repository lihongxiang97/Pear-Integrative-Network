import pandas as pd
import sys
df = pd.read_csv(sys.argv[1],sep='\t')
d = df.iloc[:,:-4].set_index('Gene_id')
for i in d.columns:
    sum = d[i].sum(axis=0)
    d[i] = d[i]/sum*1000000
d.to_csv(sys.argv[2],sep='\t')
