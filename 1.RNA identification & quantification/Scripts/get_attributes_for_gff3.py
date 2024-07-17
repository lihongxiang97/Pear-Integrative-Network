import sys
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Generate GFF3 file for circRNAs from BED and GTF files.')
    parser.add_argument('bed_file', help='Input BED file containing circRNA information')
    parser.add_argument('gtf_list', help='File containing a list of GTF files for each sample')
    parser.add_argument('output_gff3', help='Output GFF3 file for circRNAs')

    return parser.parse_args()

def main():
    args = parse_args()

    gff3 = defaultdict(dict)

    with open(args.bed_file) as f:
        for line in f:
            line = line.strip().split()
            gff3[line[3]]['Chr'] = line[0]
            gff3[line[3]]['Start'] = line[1]
            gff3[line[3]]['End'] = line[2]
            gff3[line[3]]['dot'] = line[4]
            gff3[line[3]]['Stand'] = line[5]

    gtf_list = []

    with open(args.gtf_list) as f:
        for line in f:
            gtf_list.append(line.strip())

    for i in gtf_list:
        with open(i) as f:
            for line in f:
                if not line.startswith('#'):
                    line = line.strip().split('\t')
                    id = line[8].split('; ')[0].split(' ')[1].strip('"')
                    type = line[8].split('; ')[1].split(' ')[1].strip('"')
                    if 'gene_id' in line[8]:
                        gene_id = line[8].split(';')[5].split(' ')[2].strip('"')
                    else:
                        gene_id = 'None'
                    gff3[id]['Type'] = type
                    gff3[id]['Gene_id'] = gene_id

    n = 1
    with open(args.output_gff3, 'w') as out:
        for key in gff3:
            if 'Gene_id' in gff3[key] and gff3[key]['Gene_id'] != 'None':
                print(gff3[key]['Chr'], 'CIRI', 'circRNA', gff3[key]['Start'], gff3[key]['End'], gff3[key]['dot'],
                      gff3[key]['Stand'], '.',
                      f'ID=Pbr_circRNA{n};Circ_type={gff3[key]["Type"]};Gene_id={gff3[key]["Gene_id"]};', sep='\t', file=out)
            else:
                print(gff3[key]['Chr'], 'CIRI', 'circRNA', gff3[key]['Start'], gff3[key]['End'], gff3[key]['dot'],
                      gff3[key]['Stand'], '.', f'ID=Pbr_circRNA{n};', sep='\t', file=out)
            n += 1

if __name__ == "__main__":
    main()
