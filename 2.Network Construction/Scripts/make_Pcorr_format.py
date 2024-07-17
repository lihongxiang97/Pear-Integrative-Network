import sys
description = {}
with open(sys.argv[1]) as f:
    for lines in f:
        if not lines.startswith('#'):
            line = lines.strip().split('\t')
            description[line[0]] = line[7]
out = open(sys.argv[3],'w')
with open(sys.argv[2]) as f:
    for lines in f.readlines()[1:]:
        line = lines.strip().split(',')
        if line[0] in description and line[1] in description:
            print(line[0], '-', description[line[0]], line[1], '-', description[line[1]], 'Pcorr', '-', '-', sep='\t',
                  file=out)
        elif line[0] in description and line[1] not in description:
            print(line[0], '-', description[line[0]], line[1], '-', '-', 'Pcorr', '-', '-', sep='\t',
                  file=out)
        elif line[0] not in description and line[1] in description:
            print(line[0], '-', '-', line[1], '-', description[line[1]], 'Pcorr', '-', '-', sep='\t',
                  file=out)
        elif line[0] not in description and line[1] not in description:
            print(line[0], '-', '-', line[1], '-', '-', 'Pcorr', '-', '-', sep='\t',
                  file=out)
