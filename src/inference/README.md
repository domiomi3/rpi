# Instructions to get output probabilities
Tested with Python 3.9.6
MacOS/ Linux


# Install requirements
1. create a venv
```
python -m venv venv
source venv/bin/activate
```
2. install pytorch
(maybe another version is required due to CUDA refer: https://pytorch.org/get-started/locally/)
```
pip3 install torch torchvision torchaudio
```

3. Install left python requirements
```
pip3 install requirements.txt
```

4. Download required files for ESM2 Model (refer: )
```
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt
# TODO: another file also required
```

1. Provide a csv file which contains RNA-protein pairs
Each row must contain a RNA and a protein sequence (col names: rna_sequence, protein_sequence)
   (see example.csv)
Script creates unique IDs for proteins and RNAs and stores it into a parquet file
2. Create Embeddings for protein sequences