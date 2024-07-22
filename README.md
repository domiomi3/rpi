# RNA-Protein Interaction Prediction

## Install environment
```
conda create -n rpi python=3.8 pip
pip install -r requirements.txt
```
## Install ESM-2
```
pip install git+https://github.com/facebookresearch/esm.git 
```
## Install RNA-FM
```
pip install rna-fm
```
## Infer RPIembeddor
```
python src/inference.py
```
For protein sequence, available alphabet includes ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'] and for RNA sequences ['A', 'C', 'G', 'U']. 