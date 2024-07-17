#Quality control of raw data
##114 RNA-seq of pear fruits from public databases
'''
fastp -i input_1.fastq.gz -I input_2.fastq.gz -o out_1.fastq.gz -O out_2.fastq.gz
'''
##circRNA-seq (The released data is clean data without adapters and low-quality sequences.)
Deduplication using fastuniq
'''
fastuniq -i t12 -t q -o 02.fastuniq/circ_DAF85-2.uq.R1.fq -p 02.fastuniq/circ_DAF85-2.uq.R2.fq
'''
t12 is the location of input file, like:
01.CleanData/circ_DAF85-2.R1.fq
01.CleanData/circ_DAF85-2.R2.fq