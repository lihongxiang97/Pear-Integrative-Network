import sys
d = {}
with open(sys.argv[1]) as f:
    for lines in f.readlines()[1:]:
        line = lines.strip().split('\t')
        edge = (line[0],line[3])
        d[edge] = [line[2],line[5]]
with open(sys.argv[2]) as f:
    for lines in f.readlines()[1:]:
        line = lines.strip().split('\t')
        if (line[0],line[3]) not in d and (line[3],line[0]) not in d:
            print(lines.strip())
        elif (line[0],line[3]) in d:
            print(line[0], line[1], d[(line[0],line[3])][0], line[3], line[4], d[(line[0],line[3])][1],
                  line[6] + ' & Pcorr', line[7], line[8], sep='\t')
            del d[(line[0],line[3])]
        elif (line[3],line[0]) in d:
            print(line[0], line[1], d[(line[3],line[0])][0], line[3], line[4], d[(line[3],line[0])][1],
                  line[6] + ' & Pcorr', line[7], line[8], sep='\t')
            del d[(line[3],line[0])]
for key,value in d.items():
    print(key[0],'-',value[0],key[1],'-',value[1],'Pcorr','-','-',sep='\t')
