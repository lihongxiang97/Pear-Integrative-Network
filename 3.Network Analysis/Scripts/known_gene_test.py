import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns

known_gene = pd.read_csv('known_gene.id')
network = pd.read_csv('Integrative.csv')
node = pd.read_csv('node.id')

def extract_subnetwork(network_df, ids):
    return network_df[network_df['source'].isin(ids) | network_df['target'].isin(ids)]

def calculate_ratio(subnetwork, remaining_genes):
    # 确保每个基因只计算一次
    unique_genes_in_subnetwork = set(subnetwork['source']).union(set(subnetwork['target']))
    count = len(unique_genes_in_subnetwork.intersection(set(remaining_genes)))
    return count / len(remaining_genes)


# 初始化比例列表
ratios_known_gene = []
ratios_node = []

# 随机实验重复次数
num_repetitions = 1000

# 主循环
for _ in range(num_repetitions):
    selected_known_gene = known_gene['Gene'].sample(36, replace=False)
    selected_node = node['Gene'].sample(36, replace=False)
    remaining_genes = known_gene[~known_gene['Gene'].isin(selected_known_gene)]['Gene']
    subnetwork_known_gene = extract_subnetwork(network, selected_known_gene)
    subnetwork_node = extract_subnetwork(network, selected_node)
    ratio_known_gene = calculate_ratio(subnetwork_known_gene, remaining_genes)
    ratio_node = calculate_ratio(subnetwork_node, remaining_genes)
    ratios_known_gene.append(ratio_known_gene)
    ratios_node.append(ratio_node)

data = {
    'Ratios': ratios_known_gene + ratios_node,
    'Label': ['Known Gene'] * len(ratios_known_gene) + ['Random Gene'] * len(ratios_node)
}
result_df = pd.DataFrame(data)
# 保存到CSV文件
result_df.to_csv('ratios_and_labels.csv', index=False)

# t检验
t_stat, p_value = ttest_ind(ratios_known_gene, ratios_node)

# 绘制小提琴图
plt.figure(figsize=(8, 6))
sns.violinplot(x='Label', y='Ratios', data=data, palette='muted')
# 添加显著性标记
significance_level = 0.0001
if p_value < significance_level:
    y_max = max(data['Ratios'])
    y, h, col = y_max + 0.05, 0.05, 'k'
    plt.plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text(0.5, y+h, "****", ha='center', va='bottom', color=col, fontsize=14)
plt.title('Comparison of ratios with network prediction accuracy')
plt.ylabel('Ratio')
plt.savefig('known_gene_test.pdf',dpi=300)
