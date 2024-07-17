import sys
if len(sys.argv) == 2:
    with open(sys.argv[1]) as f:
        for lines in f:
            line = lines.strip().split()
            if len(line) == 2:
                gene_id_list = line[1].split(',')
                for i in gene_id_list:
                    print(line[0],i,'1','CicrRNA',sep='\t')
else:
    print('Usage: python tran_circRNA_mRNA_interact.py circRNA-mRNA > output')
