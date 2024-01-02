## Install requirements
rna-fm package already there
```
pip install -r requirements.txt 
```

## Store RNA sequences
Store RNA sequences into dataframe following instructions on dataset.ipynb

## Create Embeddings
using script create_rna-fm_embeddings.py
e.g. 
```
python3 create_rna-fm_embeddings.py --enable-cuda=True --rna-path=unique_RNAs.parquet
```

## Convert Embeddings
Required for the usage of PandasInMemory Dataloader which preloads ALL embeddings at once.
```
python3 convert_rna-fm_embeddings.py
```
Additionally check if the create embedding is correct. 
Embeddings have to be on the right array position.
```
python3 check_rna-fm_embeddings.py --embeddings-path=rna_embeddings/ --rna_path=unique_rnas.parquet
```