# RNA-Protein Interaction Prediction
## Directories
    .
    ├── cluster_experiments     # SBATCH files (templates) to run experiments on KI-SLURM cluster
    ├── external_databases      # Scripts for all external databases needed to fetch RNA/protein sequences
    ├───── Ensembl              # Notebook to scrape data from Ensembl (Protein)
    ├───── miRBase              # Notebook to scrape data from miRBase (RNA)
    ├───── NCBI                 # Notebook to scrape data from NCBI (RNA & Protein)
    ├───── NONCODE              # Notebook to scrape data from NONCODE (RNA)
    ├───── UniProt              # Notebook to scraoe data from UniProt (Protein)
    ├── reports                 # Notebooks to analyze produced dataset
    ├── scripts                 # Notebooks and Scripts to create annotated dataset based on RNAIner
    ├───── annotate             # Notebooks to annotate cluster & RNA-family information to sequences
    ├───── embeddings           # Scripts to create all required embeddings for runnning experiments
    ├── experiments             # Scripts to run experiments with provided dataset
    ├───── optimization         # Scripts to run random search & SMAC to optimize the models`s hyperparameters
    ├───── train                # Scripts to train our model with different embeddings (without HPO)
    ├── inference               # (Not finished/ not ready) Scripts to run inference on unseen test data
    ├── reports                 # Scripts to analyze results of experiments
    ├── src                     # Modules & DataLoaders of our model
    ├───── data                 # DataLoader
    ├───── models               # Modules/Model
    ├── requirements.txt        # Packages required to run scripts (for model)
    └── README.md               # This file :-)

## Getting started (reproduction of thesis results)
```
conda create -n rpi python=3.8 pip
pip install -r requirements.txt
```
### Preparation: Download files
Copy files from cluster (KI-SLURM)
```
/data/datasets/RNAProteinInteractions/
```

Get RNAInter
```
wget http://www.rnainter.org/raidMedia/download/Download_data_RP.tar.gz
tar -xf Download_data_RP.tar.gz
rm Download_data_RP.tar.gz
```

### Step 1: Restrict sequence lengths
follow instruction on [dataset/scripts/annotate/export_sequences.ipynb](https://github.com/automl-private/RPI/blob/main/dataset/scripts/annotate/export_sequences.ipynb) 
to limit RNA & Protein sequence lengths. 
Store sequences to parquet file (rna_sequences_short.parquet, protein_sequences_short.parquet)

## Step 2: Annotate sequences with cluster information
run script [dataset/scripts/annotate/cluster_sequences.py](https://github.com/automl-private/RPI/blob/main/dataset/scripts/annotate/cluster_sequences.py).
Store annotated sequences to parquet file (rna_sequences_cluster.parquet, protein_sequences_clusters.parquet)

## Step 3: Annotate RNA sequences with RNA family information
run script [dataset/scripts/annotate/get_RNA_family.py](https://github.com/automl-private/RPI/blob/main/dataset/scripts/annotate/get_RNA_family.py).
Store annotated sequences to parquet file (rna_sequences_short_families.parquet)

## Step 4: Creating Dataset for Training
Combining all annotated information, creating negative interactions and splits using [dataset/scripts/dataset.ipynb](https://github.com/automl-private/RPI/blob/main/dataset/scripts/dataset.ipynb).

## Step 5: Creating ESM-embeddings
Follow instructions on [dataset/scripts/embeddings/ESM/README.md](https://github.com/automl-private/RPI/blob/main/dataset/scripts/embeddings/ESM/README.md)

## Step 6: Creating RNA-FM embeddings
Follow instructions on [dataset/scripts/embeddings/RNA-FM/README.md](https://github.com/automl-private/RPI/blob/main/dataset/scripts/embeddings/RNA-FM/README.md)

## Step 7: Creating RNAFormer embeddings (optional)
Follow instructions on [dataset/scripts/embeddings/RNAFormer/README.md](https://github.com/automl-private/RPI/blob/main/dataset/scripts/embeddings/RNAFormer/README.md)

## Step 8: Run final evaluation (using RNA-FM & ESM embeddings)
run script [experiments/train/train_rna-fm_random_split.py](https://github.com/automl-private/RPI/blob/main/experiments/train/train_rna-fm_random_split.py) with stated hyperparameters 
```
python train_rna-fm_random_split.py 
--accelerator gpu 
--devices 1 
--wandb 
--num-encoder-layers 1 
--max-epochs 150 
--num-dataloader-workers 8 
--batch-size 8 
--d-model 20 
--n-head 2 
--dim-feedforward 20 
--dropout 0.21759085167606332 
--weight-decay 0.00022637229697395497 
--key-padding-mask 
--lr-init 0.00001923730509654649 
--dataloader-type PandasInMemory 
--protein-embeddings-path dataset/results/protein_embeddings.npy 
--rna-embeddings-path dataset/results/rna_embeddings.npy 
--db-file-train dataset/results/final_train_set.parquet 
--db-file-valid dataset/results/final_test_set_random.parquet 
--seed=0
```
