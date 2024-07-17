import sys
d = {}
with open(sys.argv[1]) as f:
    for lines in f.readlines():
        line = lines.strip().split()
        d[line[0]] = line[1]
with open(sys.argv[2]) as f:
    for lines in f.readlines():
        line = lines.strip().split()
        if line[0] in d and line[1] in d:
            if d[line[0]] != d[line[1]]:
                print(d[line[0]],line[0],d[line[1]],line[1],sep='\t')
