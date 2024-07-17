import sys
if len(sys.argv) != 3:
    print('Usage: python cal_enriched_ratio.py gene_pairs module_file')
else:
    l = []
    with open(sys.argv[2]) as f:
        for lines in f:
            line = lines.strip().split()
            l.append(line)
    all_pairs = 0
    pairs_in_same_module = 0
    with open(sys.argv[1]) as f:
        for lines in f:
            all_pairs += 1
            line = lines.strip().split()
            for i in l:
                if line[0] in i and line[1] in i:
                    pairs_in_same_module += 1
    ratio = pairs_in_same_module/all_pairs
    print(ratio)

