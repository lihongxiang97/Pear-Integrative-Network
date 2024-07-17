import csv
import argparse

# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='Identify orthologs between Arabidopsis thaliana and Pyrus bretschneideri.')
    parser.add_argument('--orthogroups', required=True, help='Path to the Orthogroups file，eg,Ath__v__Pbr.tsv')
    parser.add_argument('--blast', required=True, help='Path to the BLAST file')
    parser.add_argument('--output', required=True, help='Path to the output file')
    parser.add_argument('--similarity', type=float, default=30.0, help='Similarity threshold for BLAST matches, default is 60')
    parser.add_argument('--score', type=float, default=100, help='Score threshold for BLAST matches, default is 100')
    return parser.parse_args()

# 读取 Orthogroup 文件
def read_orthogroups(file_path):
    orthogroups = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            group_id = row[0]
            ath_genes = row[1].split(', ')
            pbr_genes = row[2].split(', ')
            orthogroups[group_id] = {'Ath': ath_genes, 'Pbr': pbr_genes}
    return orthogroups

# 读取 BLAST 文件
def read_blast(file_path, similarity_threshold, score_threshold):
    blast_pairs = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            ath_gene = row[0]
            pbr_gene = row[1]
            similarity = float(row[2])
            score = float(row[11])
            if ath_gene not in blast_pairs or blast_pairs[ath_gene]['score'] < score:
                if similarity >= similarity_threshold and score >= score_threshold:
                    blast_pairs[ath_gene] = {'pbr_gene': pbr_gene, 'similarity': similarity, 'score': score}
    return blast_pairs

# 鉴定直系同源基因
def identify_orthologs(orthogroups, blast_pairs):
    orthologs = {}
    for group in orthogroups.values():
        for ath_gene in group['Ath']:
            if ath_gene in blast_pairs:
                pbr_gene = blast_pairs[ath_gene]['pbr_gene']
                orthologs[ath_gene] = pbr_gene
    return orthologs

# 主函数
def main():
    args = parse_arguments()

    orthogroups = read_orthogroups(args.orthogroups)
    blast_pairs = read_blast(args.blast, args.similarity, args.score)
    orthologs = identify_orthologs(orthogroups, blast_pairs)

    with open(args.output, 'w') as file:
        for ath_gene, pbr_gene in orthologs.items():
            file.write(f"{ath_gene}\t{pbr_gene}\n")

if __name__ == "__main__":
    main()
