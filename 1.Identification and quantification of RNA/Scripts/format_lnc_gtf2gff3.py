import argparse

def gtf_to_gff3(input_file, output_file, abb):
    lncRNA_dict = {}
    transcript_dict = {}
    exon_dict = {}
    with open(input_file, 'r') as gtf, open(output_file, 'w') as gff3:
        for lines in gtf:
            if lines.startswith('#'):
                # Skip comments in the input GTF file
                continue

            line = lines.strip().split('\t')
            lncRNA_id = line[8].split(';')[0].split(' ')[1].strip('"')
            transcript_id = line[8].split(';')[1].split(' ')[2].strip('"')
            exon_id = line[8].split(';')[1].split(' ')[2].strip('"') + '_' + line[8].split(';')[2].split(' ')[2].strip('"')

            exon_dict[exon_id] = [line[0],line[3],line[4],line[5],line[6],line[7]]
            if transcript_id not in transcript_dict:
                transcript_dict[transcript_id] = [exon_id]
            else:
                transcript_dict[transcript_id].append(exon_id)
            if lncRNA_id not in lncRNA_dict:
                lncRNA_dict[lncRNA_id] = [transcript_id]
            else:
                lncRNA_dict[lncRNA_id].append(transcript_id)


        lncRNA_number = 1
        for lncRNA_id,transcript_list in lncRNA_dict.items():
            chr = ''
            start = []
            end = []
            six = ''
            seven = ''
            eight = ''
            for transcript_id in transcript_list:
                for exon_id in transcript_dict[transcript_id]:
                    attributes = exon_dict[exon_id]
                    chr = attributes[0]
                    start.append(int(attributes[1]))
                    end.append(int(attributes[2]))
                    six = attributes[3]
                    seven = attributes[4]
                    eight = attributes[5]
            print('\t'.join([chr, 'Cufflinks', 'lncRNA', str(min(start)), str(max(end)), six, seven, eight]),'\t',
                  'ID=' + abb + '_lncRNA' + str(lncRNA_number) + ';', sep='', file=gff3)

            transcript_number = 1
            for transcript_id in transcript_list:
                chr = ''
                start = []
                end = []
                six = ''
                seven = ''
                eight = ''
                for exon_id in transcript_dict[transcript_id]:
                    attributes = exon_dict[exon_id]
                    chr = attributes[0]
                    start.append(int(attributes[1]))
                    end.append(int(attributes[2]))
                    six = attributes[3]
                    seven = attributes[4]
                    eight = attributes[5]
                print('\t'.join([chr, 'Cufflinks', 'transcript', str(min(start)), str(max(end)), six, seven, eight]),'\t',
                      'ID=' + abb + '_lncRNA' + str(lncRNA_number) + '_transcript' + str(transcript_number) + ';',
                      'Parent=' + abb + '_lncRNA' + str(lncRNA_number) + ';', sep='', file=gff3)

                exon_number = 1
                for exon_id in transcript_dict[transcript_id]:
                    attributes = exon_dict[exon_id]

                    print('\t'.join([attributes[0], 'Cufflinks', 'exon', attributes[1], attributes[2], attributes[3], attributes[4], attributes[5]]), '\t',
                      'ID=' + abb + '_lncRNA' + str(lncRNA_number) + '_transcript' + str(transcript_number) + '_exon' + str(exon_number) +';',
                      'Parent=' + abb + '_lncRNA' + str(lncRNA_number) + '_transcript' + str(transcript_number) + ';', sep='', file=gff3)
                    exon_number += 1
                transcript_number += 1
            lncRNA_number += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GTF to GFF3 and add lncRNA lines.")
    parser.add_argument("input_file", help="Input GTF file")
    parser.add_argument("output_file", help="Output GFF3 file")
    parser.add_argument("abb", help="Your species abbreviation")

    args = parser.parse_args()

    gtf_to_gff3(args.input_file, args.output_file, args.abb)
