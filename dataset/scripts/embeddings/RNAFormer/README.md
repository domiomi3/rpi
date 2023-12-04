## Install requirements
Following instructions from authors of RNAFormer
https://github.com/automl/RNAformer#installation
Also install additional requirements
```
conda activate rnafenv
pip install -r requirements.txt
```
## Store RNA sequences
Store RNA sequences into dataframe following instructions on dataset.ipynb

## Create Embeddings
using script create_RNAFormer_embeddings.py
e.g. 
```
python3 create_RNAFormer_embeddings.py --enable-cuda --rna-path=unique_rnas.df
```

## Convert Embeddings
Required for the usage of PandasInMemory Dataloader which preloads ALL embeddings at once.
```
python3 convert_RNAFormer_embeddings.py
```