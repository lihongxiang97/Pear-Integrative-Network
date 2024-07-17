with open("gene_id") as file:
    gene_ids = [line.strip() for line in file]
with open("gene_pairs",'w') as output_file:
    # 遍历所有基因ID
    for i in range(len(gene_ids)):
        for j in range(i+1, len(gene_ids)):
            output_file.write(f"{gene_ids[i]} {gene_ids[j]}\n")
