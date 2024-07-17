import argparse
parser = argparse.ArgumentParser(description="It will add database source of interaction.")
parser.add_argument("-A","--AI",required=True,help="AI_interaction.txt")
parser.add_argument("-B","--BIOGRID",required=True,help="BIOGRID_interaction.txt")
parser.add_argument("-T1","--Tair1",required=True,help="TairPI1.0.txt")
parser.add_argument("-T2","--Tair2",required=True,help="TairPI2.0.txt")
parser.add_argument("-P","--PbrI",required=True,help="Pbr_interactions_pro.txt")

args = parser.parse_args()

def make_d(file):
    d = {}
    with open(file) as f:
        for lines in f.readlines():
            line = lines.strip().split()
            if line[0] not in d:
                d[line[0]] = [line[1]]
            else:
                d[line[0]].append(line[1])
    return d

AI = make_d(args.AI)
BIOGRID = make_d(args.BIOGRID)
T1 = make_d(args.Tair1)
T2 = make_d(args.Tair2)

with open(args.PbrI) as f:
    for lines in f.readlines():
        line = lines.strip().split()
        label = []
        if line[1] in AI:
            if line[3]  in AI[line[1]]:
                label.append('AthInteractome')

        if line[1] in BIOGRID:
            if line[3] in BIOGRID[line[1]]:
                label.append('BIOGRID')

        if line[1] in T1:
            if line[3] in T1[line[1]]:
                label.append('TairPI1.0')

        if line[1] in T2:
            if line[3] in T2[line[1]]:
                label.append('TairPI2.0')

        print(line[0],line[1],line[2],line[3],' & '.join(label),sep='\t')
