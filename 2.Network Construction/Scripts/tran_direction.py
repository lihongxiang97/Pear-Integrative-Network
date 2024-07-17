import sys
l = []
with open(sys.argv[1]) as f:
    for lines in f:
        line = lines.strip().split()
        l.append(line[0])
with open(sys.argv[2]) as f:
    for lines in f:
        line = lines.strip().split(',')
        if line[1] in l:
            print(line[1],line[0],line[2],line[3],sep=',')
        else:
            print(lines.strip())
