import argparse
import random
import os

import torch

import pandas as pd
import numpy as np

from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Optional


class RNAInterActionsPandasInMemory(Dataset):
    """
    Loads Pandas Dataframe and reads embeddings from storage when needed.
    Recommended for using on cluster. Be aware that its usage needs a lot of system memory (RAM) since all embeddings
    have to be preloaded into memory.
    """
    def __init__(self,
                 rna_embeddings: np.array,
                 protein_embeddings: np.array,
                 train_set_path: str
                 ):
        self.train_set = pd.read_parquet(train_set_path, engine='pyarrow')
        self.train_set = self.train_set.assign(row_number=range(len(self.train_set)))

        self.protein_embeddings = protein_embeddings
        self.rna_embeddings = rna_embeddings
        self.length = self.train_set.shape[0]

    @staticmethod
    def pre_load_rna_embeddings(
            rna_embeddings_path: str
    ):
        print("Loading RNA embeddings...")
        start = time()
        rna_embeddings = np.load(rna_embeddings_path)
        print(f"Loaded RNA embeddings into RAM in {round(time() - start, 2)} seconds")
        return rna_embeddings

    @staticmethod
    def pre_load_protein_embeddings(
            protein_embeddings_path: str
    ):
        print("Loading Protein embeddings...")
        start = time()
        protein_embeddings = np.load(protein_embeddings_path)
        print(f"Loaded Protein embeddings into RAM in {round(time() - start, 2)} seconds")
        return protein_embeddings

    @staticmethod
    def pre_load_embeddings(
            rna_embeddings_path: str,
            protein_embeddings_path: str, ):
        return (RNAInterActionsPandasInMemory.pre_load_rna_embeddings(rna_embeddings_path),
                RNAInterActionsPandasInMemory.pre_load_protein_embeddings(protein_embeddings_path))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):

        # Get entry information
        row = self.train_set[self.train_set['row_number'] == index][
            ['Sequence_1_emb_ID', 'Sequence_2_emb_ID', 'interaction', 'row_number']]
        assert row.shape[0] == 1

        seq_1_emb_ID, seq_2_emb_ID, interaction, row_number = row.values.tolist()[0]

        interaction = float(interaction)

        # Get already padded embeddings
        padded_seq_1_embed = self.rna_embeddings[seq_1_emb_ID]
        padded_seq_2_embed = self.protein_embeddings[seq_2_emb_ID]

        return padded_seq_1_embed, padded_seq_2_embed, interaction, row_number


def get_dataloader(
        dataset_path: str,
        rna_embeddings_path: str,
        protein_embeddings_path: str,
        split_set_size: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
    """
    Loads interaction dataset from .parquet file and returns two DataLoader objects.
    When using for training, provide split_set_size and seed to enable random split
    between training and validation sets.
    When using for testing, skip the optional arguments.

    Args:
    - dataset_path (str): Path to the dataset file.
    - rna_embeddings_path (str): Path to the RNA embeddings file.
    - protein_embeddings_path (str): Path to the protein embeddings file.
    - split_set_size (float): Size of the validation set. If None, no splitting is done.
    - seed (int): Seed for reproducibility of the split.

    Returns:
    train_dataloader (DataLoader): DataLoader object for the training set.
    valid_dataloader (DataLoader): DataLoader object for the validation set. If split_set_size is None, returns None.
    """
    
    if seed:
        set_seed(seed)
    
    rna_embeddings, protein_embeddings = RNAInterActionsPandasInMemory.pre_load_embeddings(
        rna_embeddings_path,
        protein_embeddings_path
    )
    dataset = RNAInterActionsPandasInMemory(
        rna_embeddings,
        protein_embeddings,
        dataset_path,
    )

    if split_set_size is None:
        return DataLoader(dataset, shuffle=False, **kwargs), None
    else: 
        train_set, valid_set = random_split(dataset, [1-split_set_size, split_set_size])
        
        assert train_set is not None, 'train_set is None. Check if the path is correct.' 
        assert valid_set is not None, 'valid_set is None. Check if the path is correct.'

        return DataLoader(train_set, shuffle=True, **kwargs), DataLoader(valid_set, shuffle=False, **kwargs)


def set_seed(seed):
    """
    Set the seed for reproducibility in random operations.

    Args:
    seed (int): The seed value to be set for all relevant libraries.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # Numpy library
    os.environ['PYTHONHASHSEED'] = str(seed)  # Environment variable

    # For PyTorch 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False