import sys

file_list = []
with open(sys.argv[1]) as f:
    for lines in f:
        file_list.append(lines.strip())

d = {}
sample_list = []
# 逐个读取文件并合并到结果DataFrame中
for file in file_list:
    # 从文件名中提取样本名
    sample_name = file.split('_')[1]
    sample_list.append(sample_name)

    with open(file) as f:
        for lines in f.readlines()[1:]:
            line = lines.strip().split()
            if line[0] not in d:
                d[line[0]] = [line[8]]
            else:
                d[line[0]].append(line[8])
out = open(sys.argv[2],'w')
print('Gene_ID','\t'.join(sample_list),sep='\t',file=out)
for key,value in d.items():
    print(key,'\t'.join(value),sep='\t',file=out)
