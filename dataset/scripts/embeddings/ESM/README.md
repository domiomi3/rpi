## Install requirements
https://github.com/facebookresearch/esm#getting-started-with-this-repo-
```
pip install -r requirements
```
https://github.com/facebookresearch/esm#available-models-and-datasets- 
Download model esm2_t30_150M_UR50D
## Store protein sequences
Store protein sequences into dataframe following instructions on dataset.ipynb

## Create Embeddings
using script create_esm_embeddings.py
e.g. 
```
python3 create_esm_embeddings.py --enable-cuda --model-name=esm2_t30_150M_UR50D --protein-path=unique_proteins.df --repr-layer=30
```

## Convert Embeddings
Required for the usage of PandasInMemory Dataloader which preloads ALL embeddings at once.
```
python3 convert_esm_embeddings.py
```