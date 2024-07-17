import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
# 设置全局字体大小
mpl.rcParams.update({'font.size': 15})
# 读取数据到DataFrame
data = pd.read_csv("all.sd", sep='\t', header=None)
data.columns = ['Type', 'sd', 'Network']
type_order = ["Tandem", "Proximal", "WGD", "Transposed", "Dispersed", "Non-paralogous"]
# Get unique Type values in the specified order
data['Type'] = pd.Categorical(data['Type'], categories=type_order, ordered=True)
# 设置调色板
sns.set_palette(['#BEBAB9', '#C47070', '#92B4A7', '#E1DD8F'])

# 绘制箱型图
plt.figure(figsize=(10, 6))
sns.boxplot(x='Type', y='sd', hue='Network', data=data, fliersize=1)
plt.ylim(0.8,7.5)
plt.xlabel('')
plt.xticks(rotation=15,fontsize=15)
plt.ylabel('SD',fontsize=20)
plt.legend(title='', prop={'size': 9})
plt.grid(False)  # 移除网格线
plt.show()










