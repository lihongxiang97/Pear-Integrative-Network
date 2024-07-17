# Network Analysis
This section is the detailed steps of network analysis.
## 1.Network attribute calculation
```
python NetInfo.py network.csv nodeinfo.txt sd.txt transtivity.txt 
```
## 2.Hub genes detection
```
python get_hub_genes.py nodeinfo.txt hub_gene
```
## 3.Network clustering using MCL
```
#mcl 14-137
mcl network.txt --abc -o module_network
```
## 4.Network clustering using MCL
```
#mcl 14-137
mcl network.txt --abc -o network_module
python set_network_module.py -m network_module -o network.module.csv
```
## 5.Known gene test
```
python known_gene_test.py
```





