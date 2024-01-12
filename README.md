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
Copy files from cluster (KI-SLURM, hidden dir) 
```
/data/datasets/RNAProteinInteractions/
```

### Step 1: Restrict sequence lengths
follow instruction on [dataset/scripts/annotate/export_sequences.ipynb](https://github.com/automl-private/RPI/blob/main/dataset/scripts/annotate/export_sequences.ipynb) 
to limit RNA & Protein sequence lengths. PROTEIN_LEN and RNA_LEN adjustable
Store sequences to parquet file (rna_sequences_short.parquet, protein_sequences_short.parquet, unique_rna_sequences.pickle -> same as rna_short_sequences?)

## Step 2: Annotate sequences with cluster information
cd-hit installation;
conda install -c bioconda cd-hit as per [Installation](https://github.com/weizhongli/cdhit/wiki/2.-Installation)

run script [dataset/scripts/annotate/cluster_sequences.py](https://github.com/automl-private/RPI/blob/main/dataset/scripts/annotate/cluster_sequences.py).
Store annotated sequences to parquet file (rna_sequences_clusters.parquet, protein_sequences_clusters.parquet)

## Step 3 Pfam
every time we run pfam scan, we need to redownload all files AND extract binaries because they get removed after each scan (weird, need to look into it)
export PATH=/work/dlclarge1/matusd-rpi/hmmer3:$PATH
export PATH=/work/dlclarge1/matusd-rpi/bin:$PATH

export PERL5LIB=/work/dlclarge1/matusd-rpi/RPI/data/pfam/PfamScan:$PERL5LIB
export PERL5LIB=/work/dlclarge1/matusd-rpi/lib/perl5:$PERL5LIB


those not needed??
write :use lib "data/pfam/PfamScan/Bio/Pfam/Scan/PfamScan";
in pfam_scan.pl after comments

cpan
cpan Moose

if it fails due to permissions error in /usr/local/bin

update env vars

run cpan Moose
sometimes because of changing install path to local dir, dpeendencies are not handled 
then install them manually like this

cpan CPAN::Meta::Check etc
alternatively, check if cpan has correct drectives
enter
o conf makepl_arg INSTALL_BASE=/work/dlclarge1/matusd-rpi/
o conf mbuildpl_arg --install_base /work/dlclarge1/matusd-rpi/
o conf commit

and reattempt installation

install bioperl
cpan Bio::Perl

export path


change perl path in the pfam_scan.pl

## Step 3: Annotate RNA sequences with RNA family information
conda install -c bioconda infernal
run script [dataset/scripts/annotate/get_RNA_family.py](https://github.com/automl-private/RPI/blob/main/dataset/scripts/annotate/get_RNA_family.py).
Store annotated sequences to parquet file (rna_sequences_short_families.parquet)

## Step 4: Creating Dataset for Training
Combining all annotated information, creating negative interactions and splits using [dataset/scripts/dataset.ipynb](https://github.com/automl-private/RPI/blob/main/dataset/scripts/dataset.ipynb).

## Step 5: Creating ESM-embeddings
Follow instructions on [dataset/scripts/embeddings/ESM/README.md](https://github.com/automl-private/RPI/blob/main/dataset/scripts/embeddings/ESM/README.md)

## Step 6: Creating RNA-FM embeddings
Follow instructions on [dataset/scripts/embeddings/RNA-FM/README.md](https://github.com/automl-private/RPI/blob/main/dataset/scripts/embeddings/RNA-FM/README.md)

## RPI dataset
Download RPI2825 dataset from here: https://universe.bits-pilani.ac.in/goa/aduri/xRPI (data sets used in validating xRPI)

And save the RPI2825.csv in the data/rpi2825 folder

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
