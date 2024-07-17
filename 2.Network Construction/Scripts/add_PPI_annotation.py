import argparse
parser = argparse.ArgumentParser(description="It will add annotation of interaction.")
parser.add_argument("-B","--BIOGRID",required=True,help="BIOGRID.txt")
parser.add_argument("-T1","--Tair1",required=True,help="TairProteinInteraction1.0.txt")
parser.add_argument("-T2","--Tair2",required=True,help="TairProteinInteraction2.0.txt")
parser.add_argument("-P","--PbrI",required=True,help="Pbr_interactions_pro_label.txt")

args = parser.parse_args()

def read_file(file):
    d = {}
    with open(file) as f:
        for lines in f:
            line = lines.strip().split('\t')
            d[line[0]+'\t'+line[1]] = line[2:]
    return d

BIOGRID = read_file(args.BIOGRID)
T1 = read_file(args.Tair1)
T2 = read_file(args.Tair2)

with open(args.PbrI) as f:
    for lines in f:
        line = lines.strip().split('\t')
        key = line[1]+'\t'+line[3]
        source = line[4].split(' & ')
        descriptionA = []
        descriptionB = []
        publication = []
        experiment = []
        if ['AthInteractome'] == source:
            print(line[0],line[1],'-',line[2],line[3],'-',line[4],'PMCID:PMC3170756','-',sep='\t')
        elif ['BIOGRID'] == source or ['BIOGRID','TairPI2.0'] == source:
            print(line[0],line[1],BIOGRID[key][0],line[2],line[3],BIOGRID[key][1],line[4],
                  BIOGRID[key][3],BIOGRID[key][2],sep='\t')
        elif ['TairPI2.0'] == source:
            print(line[0], line[1], T2[key][0], line[2], line[3], T2[key][1], line[4],
                  '-', '-', sep='\t')
        elif ['TairPI1.0'] == source:
            print(line[0], line[1], '-', line[2], line[3], '-', line[4],
                  T1[key][1], T1[key][0], sep='\t')
        elif ['AthInteractome','BIOGRID'] == source or ['AthInteractome','BIOGRID','TairPI2.0'] == source:
            publication.append(BIOGRID[key][3])
            publication.append('PMCID:PMC3170756')
            publication = list(set(publication))
            print(line[0], line[1], BIOGRID[key][0], line[2], line[3], BIOGRID[key][1], line[4],
                  ';'.join(publication),BIOGRID[key][2] , sep='\t')
        elif ['AthInteractome','BIOGRID','TairPI1.0'] == source or ['AthInteractome','BIOGRID','TairPI1.0','TairPI2.0'] == source:
            publication.append(BIOGRID[key][3])
            publication.append('PMCID:PMC3170756')
            if len(T1[key]) == 2:
                publication.append(T1[key][1])
            elif len(T1[key]) == 1:
                publication.append('-')
            publication = list(set(publication))
            print(line[0], line[1], BIOGRID[key][0], line[2], line[3], BIOGRID[key][1], line[4],
                  ';'.join(publication), BIOGRID[key][2], sep='\t')
        elif ['AthInteractome','TairPI1.0'] == source:
            if len(T1[key]) == 2:
                publication.append(T1[key][1])
            elif len(T1[key]) == 1:
                publication.append('-')
            publication.append('PMCID:PMC3170756')
            publication = list(set(publication))
            print(line[0], line[1], '-', line[2], line[3], '-', line[4],
                  ';'.join(publication), T1[key][0], sep='\t')
        elif ['AthInteractome','TairPI1.0','TairPI2.0'] == source:
            if len(T1[key]) == 2:
                publication.append(T1[key][1])
            elif len(T1[key]) == 1:
                publication.append('-')
            publication.append('PMCID:PMC3170756')
            publication = list(set(publication))
            print(line[0], line[1], T2[key][0], line[2], line[3], T2[key][1], line[4],
                  ';'.join(publication), T1[key][0], sep='\t')
        elif ['AthInteractome','TairPI2.0'] == source:
            print(line[0], line[1], T2[key][0], line[2], line[3], T2[key][1], line[4],
                  'PMCID:PMC3170756', '-', sep='\t')
        elif ['BIOGRID','TairPI1.0'] == source or ['BIOGRID','TairPI1.0','TairPI2.0'] == source:
            publication.append(BIOGRID[key][3])
            publication.append('PMCID:PMC3170756')
            publication = list(set(publication))
            experiment.append(BIOGRID[key][2])
            experiment.append(T1[key][0])
            experiment = list(set(experiment))
            print(line[0], line[1], BIOGRID[key][0], line[2], line[3], BIOGRID[key][1], line[4],
                  ';'.join(publication), ';'.join(experiment), sep='\t')
        elif ['TairPI1.0','TairPI2.0'] == source:
            print(line[0], line[1], T2[key][0], line[2], line[3], T2[key][1], line[4],
                  T1[key][1], T1[key][0], sep='\t')
