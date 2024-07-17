import pandas as pd
df = pd.read_csv('sd.txt',sep='\t',index_col=0)
df.to_parquet('sd.parquet')
