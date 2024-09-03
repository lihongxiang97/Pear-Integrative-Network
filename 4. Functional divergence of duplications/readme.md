# Functional divergence of duplications
This section is to cluster and evaluate the functional differences of duplicate genes from a network perspective.
## 1.Detection of duplicate genes
Using Nnu as the outgroup, dupgen_finder was used to detect five types of duplicate genes.
## 2.Extraction of non-repetitive genes
Genes that do not belong to any of the categories of duplicate genes are classified as Non-paralogous genes.
## 3.Detect the ratio of pairs in the same module
```
# Input two columns of pairs and mcl clustering module files to output the proportion in the same module
python cal_enriched_ratio.py ../1.dup_gene_pairs/Pbr.transposed.pairs network_module
#Non-paralogous genes randomly pick two pairs, take 1000 pairs, and calculate the proportion of the 1000 pairs that belong to the same module
python cal_enriched_ratio_for_non_para.py ../1.dup_gene_pairs/Non-paralogous.genes network_module
#Run each type and record it in Enriched_ratio.csv, and use plot_bar.py script to draw the graph
python plot_bar.py
```
## 4.Detect SD between different types of duplicate genes
```
python get_sd.py Pbr.tandem.pairs GRN-sd.parquet Tandem >Tandem.sd
#Randomly select 1000 pairs of non-paralogous genes and extract their sd
python get_sd_for_non_para.py Non-paralogous.genes GRN-sd.parquet Non-paralogous >non-paralogous.sd
```
## 5.Identity five divergence patterns of duplicate genes
```
#Process the network edge files
a=trans_network.csv
cut -f 1,2 -d ',' $a|sed 's/,/\t/g' >`basename $a .csv`.txt
#run genecommon.py for 5 types
for i in `cat ../GRN/id.txt`;do nohup python ../genecommon.py trans_network.txt nodeinfo.txt ../../1.dup_gene_pairs/Pbr.$i.pairs >trans_$i.txt &;done
#Organize five differentiation types using dabsab.pl
#To use this script, you need to prepare id.txt in the current directory
$ cat id.txt
Dispersed
Proximal
Tandem
Transposed
WGD
###############
perl dabsab.pl GRN >GRN-type.txt
```
## 6.WGD change
```
grep wgd trans/trans-type.txt|cut -f 1,2,5 >WGD-change.txt
grep wgd prot/prot-type.txt|cut -f 5 >1
paste WGD-change.txt 1 >tmp && mv tmp WGD-change.txt
#Add head: id1\tid2\tIN\tGRN
#Remove empty rows
awk -F'\t' '$3 != "" && $4 != "" {print}' WGD-change.txt >t &&mv t WGD-change.txt
# Take the changed gene pairs and draw a picture
awk -F'\t' '$3 != $4' WGD-change.txt >t && mv t WGD-change.txt
```
