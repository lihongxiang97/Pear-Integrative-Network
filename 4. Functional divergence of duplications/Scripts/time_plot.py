import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 读取数据
data = pd.read_csv('prot-time.txt', sep='\t', header=None, names=['gene1','gene2','Type', 'Time'])

# 按照指定顺序对类别进行排序
time_order = ['After single WGD', 'After gamma WGD', 'Before gamma WGD']
data['Time'] = pd.Categorical(data['Time'], categories=time_order, ordered=True)

# 计算每个类别的频率
category_counts = data.groupby(['Type', 'Time']).size().unstack(fill_value=0)

# 计算每个类别的百分比
category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0) * 100

custom_palette = ['#66AEDB', '#8FD7BB', '#F0E49D']
sns.set_palette(custom_palette)

# 绘图
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制堆叠柱状图
category_percentages.plot(kind='bar', stacked=True, ax=ax)

# 设置标签和标题，并调整字体大小
plt.xlabel('')
plt.ylabel('Percentage (%)', fontsize=28)  # 调整字体大小


# 调整x轴标签顺序
plt.xticks(rotation=0, fontsize=30)  # 调整字体大小

# 调整y轴标签的字体大小
plt.yticks(fontsize=24)  # 调整字体大小

# 显示图例，并调整字体大小
plt.legend(title='', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=20)  # 调整字体大小

# 显示图形
plt.tight_layout()
plt.show()

