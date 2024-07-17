import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate CPM values for circRNAs from BED and GTF files.')
    parser.add_argument('bed_file', help='Input BED file containing circRNA information')
    parser.add_argument('gtf_list', help='File containing a list of GTF files for each sample')
    parser.add_argument('output_cpm', help='Output file for CPM values')

    return parser.parse_args()


args = parse_args()

# 读取bed文件为DataFrame
bed_df = pd.read_csv(args.bed_file, sep='\t', header=None, names=['Chr', 'Start', 'End', 'ID', 'Dot', 'Strand'])
bed_df.index = bed_df['ID']

# 读取每个样本的gtf文件，提取CPM值并合并到结果DataFrame
gtf_files = []

with open(args.gtf_list) as f:
    for line in f:
        gtf_files.append(line.strip())

for gtf_file in gtf_files:
    sample_name = gtf_file.split(".")[0].split('_')[1]  # 提取样本名
    gtf_df = pd.read_csv(gtf_file, sep='\t', skiprows=5, header=None,
                         names=['Chr', 'Source', 'Feature', 'Start', 'End', 'CPM', 'Strand', 'Dot', 'Attributes'])
    # 提取circRNA的ID和CPM值
    circRNA_data = gtf_df['Attributes'].copy()
    circRNA_data = pd.concat([circRNA_data,circRNA_data.str.extract(r'circ_id "(.*?)";')],axis=1)
    circRNA_data[sample_name]= circRNA_data['Attributes'].str.extract(r'bsj (\d+\.\d+);')
    circRNA_data.columns = ['Attributes','ID',sample_name]
    circRNA_data = circRNA_data[['ID', sample_name]].set_index('ID')
    # 合并到结果DataFrame
    bed_df = pd.concat([bed_df, circRNA_data], axis=1)

# 将NaN值替换为0
bed_df = bed_df.fillna(0)

#提取count列
count_cols = bed_df.columns[6:]
for i in count_cols:
    bed_df[i] = pd.to_numeric(bed_df[i])
    sum = bed_df[i].sum(axis=0)
    bed_df[i] = bed_df[i]/sum*1000000

# 输出结果
bed_df.to_csv(args.output_cpm, sep='\t', index=False)
