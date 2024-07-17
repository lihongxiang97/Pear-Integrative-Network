# Quality Control of Raw Data

## 114 RNA-seq of Pear Fruits from Public Databases

```bash
fastp -i input_1.fastq.gz -I input_2.fastq.gz -o out_1.fastq.gz -O out_2.fastq.gz
```
## circRNA-seq
The released data has been clean data without adapters and low-quality sequences.
Deduplication using fastuniq:
```bash
fastuniq -i t12 -t q -o 02.fastuniq/circ_DAF85-2.uq.R1.fq -p 02.fastuniq/circ_DAF85-2.uq.R2.fq
```
t12 is the location of input file, like:
```
01.CleanData/circ_DAF85-2.R1.fq
01.CleanData/circ_DAF85-2.R2.fq
```
## microRNA-seq
```bash
#cutadapt version 1.9.1
#quality control
for i in `cat sample_list`;do nohup cutadapt -q 20,20 -o 02.CleanData/${i}_trimmed.fq.gz 01.RawData/${i}.fq.gz >02.CleanData/$i.filter.log &;done
#remove adapters
for i in `cat sample_list`;do nohup cutadapt -a TGGAATTCTCGGGTGCCAAGG -g GTTCAGAGTTCTACAGTCCGACGATC -o 02.CleanData/${i}_trimmed_adapter.fq.gz --discard-untrimmed -e 0.05 -O 14 --no-indels 02.CleanData/${i}_trimmed.fq.gz >$i.rm_adapter.log &;done
#remove polyA
for i in `cat sample_list`;do
nohup cutadapt -a "A{10}" --no-trim --untrimmed-o 02.CleanData/${i}_trimmed_adapter_rmploya.fq.gz -m 15 -e 0 -O 10 --no-indels --max-n 0.1 02.CleanData/${i}_trimmed_adapter.fq.gz >$i.rmpolya.log &
done
```
