import argparse
def main():
    parser = argparse.ArgumentParser(description="Process input files and generate gff3 and annotation files.")
    parser.add_argument("pre_anno", help="anno file generated in 5.blast")
    parser.add_argument("pre_gff", help="gff file generated in 3.miRDP2_merge")
    parser.add_argument("final_gff3", help="Output GFF3 file")
    parser.add_argument("final_anno", help="Output annotation file")

    args = parser.parse_args()

    d = {}

    with open(args.pre_anno) as f:
        for lines in f:
            line = lines.strip().split()
            d[line[0]] = [line[1], line[2]]

    out_gff3 = open(args.final_gff3, 'w')
    out_anno = open(args.final_anno, 'w')

    with open(args.pre_gff) as gff:
        for lines in gff:
            line = lines.strip().split()
            ID, mature_seq, pre_loc, pre_seq, source_read, none = line[8].split(';')
            seq = mature_seq.split('=')[1]
            p_seq = pre_seq.split('=')[1]
            id = ID.split('=')[1]

            print('\t'.join(line[0:8]),
                  ID + ';Type=' + d[id][0] + ';' + mature_seq + ';' + pre_loc + ';' + pre_seq + ';miRBase_ID=' + d[id][
                      1] + ';',
                  sep='\t', file=out_gff3)

            print(id, line[0], line[3], line[4], line[6], d[id][0], d[id][1], seq, p_seq, sep='\t', file=out_anno)

    out_gff3.close()
    out_anno.close()


if __name__ == "__main__":
    main()
