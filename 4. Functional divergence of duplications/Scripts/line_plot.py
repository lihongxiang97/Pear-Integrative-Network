import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# 设置全局字体大小
mpl.rcParams.update({'font.size': 15})
# Read data into DataFrame
data = pd.read_csv("all.sd", sep='\t', header=None)
data.columns = ['Type', 'sd', 'Network']

# Define the order of networks
network_order = ['Integrative Network', 'Gene Regulation Network', 'Coexpression Network', 'Interactome Network']

# Define the order of types
type_order = ["Tandem", "Proximal", "WGD", "Transposed", "Dispersed", "Non-paralogous"]

# Get unique Type values in the specified order
data['Type'] = pd.Categorical(data['Type'], categories=type_order, ordered=True)

# Calculate mean and standard error for each group
mean_values = data.groupby(['Type', 'Network'])['sd'].mean().reset_index()
se_values = data.groupby(['Type', 'Network'])['sd'].sem().reset_index()

# Sort mean_values and se_values according to the specified order
mean_values = mean_values.sort_values(['Type', 'Network'])
se_values = se_values.sort_values(['Type', 'Network'])

# Define colors for each network
network_colors = ['#BEBAB9', '#C47070', '#92B4A7', '#E1DD8F']

# Plot the line plot
plt.figure(figsize=(10, 6))

for idx, network in enumerate(network_order):
    network_mean = mean_values[mean_values['Network'] == network]
    network_se = se_values[se_values['Network'] == network]

    color = network_colors[idx]
    plt.errorbar(network_mean['Type'], network_mean['sd'], yerr=network_se['sd'], marker='o',
                 label=network, color=color)

plt.xlabel('')
plt.ylabel('Mean SD',fontsize=18)
plt.xticks(rotation=15,fontsize=15)
plt.title('')
plt.legend(title='', prop={'size': 12})
plt.grid(False)
plt.ylim(1.85,3.6)
plt.show()








