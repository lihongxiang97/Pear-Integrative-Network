import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# 设置全局字体大小
mpl.rcParams.update({'font.size': 15})
# Read the CSV file
df = pd.read_csv("Enriched_ratio.csv", index_col=0)

# Transpose the DataFrame to match the new data format
df = df.T

# Convert data to percentages
df *= 100

# Extract data
categories = df.index.tolist()
enriched_ratios = df.values.tolist()

# Define colors
nature_colors = ['#BEBAB9', '#C47070', '#92B4A7', '#E1DD8F']

# Create bar plot
plt.figure(figsize=(10, 6))

bars = []
for i, category in enumerate(categories):
    bar = plt.bar([x + i * 0.2 for x in range(len(df.columns))], enriched_ratios[i], width=0.2, color=nature_colors[i], label=category)
    bars.append(bar)

# Add labels on top of bars with smaller font size
for bar_group in bars:
    for bar in bar_group:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}%", ha='center', va='bottom', fontsize=8)

# Set x-axis labels and ticks
plt.xticks([x + 0.3 for x in range(len(df.columns))], df.columns)

# Add labels and title
plt.xlabel("")
plt.ylabel("Ratio of Enriched in Same Module (%)",fontsize=16)
plt.xticks(rotation=15)
plt.legend(title='', prop={'size':11})
plt.title("")

# Set y-axis limit
plt.ylim(0, 90)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


