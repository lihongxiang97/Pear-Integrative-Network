# Machine Learning
This section contains the machine learning scripts and steps used in the research.
## 1.Prepare Python environment for machine learning
Make sure you have Python 3 and pip installed.  
The following are the version information of the Python modules required by the script：  
```
python==3.9.18  
numpy==1.26.2  
pandas==2.1.3  
scikit-learn==1.3.2  
tensorflow==2.15.0  
keras==2.15.0  
tqdm==4.66.1  
xgboost==2.0.2  
plotnine==0.12.4  
tqdm==4.66.1
```
After testing, the script can run in this environment.
## 2.Prepare positive genes and negative genes
The sample files are in the Data/1.train_gene.
## 3.Start training and testing
### Using pipline script
```
train_test_pipline.py --positive positive_gene --negative negative_gene --test test_gene --nodeinfo nodeinfo.txt --sd sd.parquet --TPM all.TPM.csv --Protein Pbr_Fruit_protein.csv --train_type Multi_transcriptomic Proteomic Fd_transcriptomic

  --positive POSITIVE   Positive gene list file
  --negative NEGATIVE   Negative gene list file
  --test TEST           test gene list file
  --nodeinfo NODEINFO   Nodeinfo file of network
  --sd SD               sd.parquet file of network,sd.parquet can transfrom from sd.txt by script 'tran_parquet.py'
  --TPM TPM             TPM file of network
  --Protein PROTEIN     Protein file of network
  --train_type {Multi_transcriptomic,Fd_transcriptomic,Proteomic} [{Multi_transcriptomic,Fd_transcriptomic,Proteomic} ...]
                        The data type you want to train, you can add one or more type to train data.
```
### Or you can use standalone scripts for training and prediction
```
#example:
Trainning_results_Fd_transcriptomic_Proteomic.py --train train.csv --test test.csv --randomnum 20 --modeldir ./model --dir ./result --gpu
```
