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
