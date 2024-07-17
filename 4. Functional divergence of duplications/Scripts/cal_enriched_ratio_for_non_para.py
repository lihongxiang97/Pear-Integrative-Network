import sys
import random
if len(sys.argv) != 3:
    print('Usage: python cal_enriched_ratio.py Non-paralogous.genes module_file')
else:
    l = []
    with open(sys.argv[2]) as f:
        for lines in f:
            line = lines.strip().split()
            l.append(line)

    non_para_genes = []
    with open(sys.argv[1]) as f:
        for lines in f:
            line = lines.strip()
            non_para_genes.append(line)
    a = 0
    for n in range(1000):
        pairs_in_same_module = 0
        for _ in range(1000):
            non_para_pairs = random.sample(non_para_genes,2)
            for i in l:
                if non_para_pairs[0] in i and non_para_pairs[1] in i:
                    pairs_in_same_module += 1
        ratio = pairs_in_same_module/1000
        a += ratio
        print(ratio)
    print(a/1000)
