import random
import os

import torch

import pandas as pd
import numpy as np

from time import time
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Optional


class RPIDataset(Dataset):
    """
    Loads Pandas Dataframe and reads embeddings from storage when needed.
    Recommended for using on cluster. Be aware that its usage needs a lot of 
    system memory (RAM) since all embeddings have to be preloaded into memory.
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
        return (RPIDataset.pre_load_rna_embeddings(rna_embeddings_path),
                RPIDataset.pre_load_protein_embeddings(protein_embeddings_path))

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


class RPIDatasetRNARand(RPIDataset):
    """
    Data Loader Class for ablation experiment.
    Replaces the RNA input with a random tensor.
    However, keeps the padded zero values and generates random values within 
    the range (min, max) of the original tensor.
    """
    def __getitem__(self, index):
        padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id = super().__getitem__(index)
        
        # Replace sequence by random
        non_zero_indices = padded_seq_1_embed.nonzero()
        random_values = (padded_seq_1_embed.min() - padded_seq_1_embed.max()) * torch.rand(
            non_zero_indices[0].shape[0]) + padded_seq_1_embed.max()
        padded_random_seq_1 = padded_seq_1_embed.copy()
        padded_random_seq_1[non_zero_indices] = random_values

        return padded_random_seq_1, padded_seq_2_embed, interacts, rna_inter_id


class RPIDatasetProteinRand(RPIDataset):
    """
    Data Loader Class for ablation experiment.
    Replaces the protein input with a random tensor.
    However, keeps the padded zero values and generates random values within 
    the range (min, max) of the original tensor.
    """
    def __getitem__(self, index):
        padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id = super().__getitem__(index)
        
        # Replace sequence by random
        non_zero_indices = padded_seq_2_embed.nonzero()
        random_values = (padded_seq_2_embed.min() - padded_seq_2_embed.max()) * torch.rand(
            non_zero_indices[0].shape[0]) + padded_seq_2_embed.max()
        padded_random_seq_2 = padded_seq_2_embed.copy()
        padded_random_seq_2[non_zero_indices] = random_values

        return padded_seq_1_embed, padded_random_seq_2, interacts, rna_inter_id


def get_dataloader(
        loader_type: str,
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
    - loader_type (str): Type of dataloader to be used. Can be one of 
    ['RPIDatasetProteinRand', 'RPIDatasetRNARand', 'PandasInMemory'].
    - dataset_path (str): Path to the dataset file.
    - rna_embeddings_path (str): Path to the RNA embeddings file.
    - protein_embeddings_path (str): Path to the protein embeddings file.
    - split_set_size (float): Size of the validation set. If None, no splitting is done.
    - seed (int): Seed for reproducibility of the split.

    Returns:
    train_dataloader (DataLoader): DataLoader object for the training set.
    valid_dataloader (DataLoader): DataLoader object for the validation set. If split_set_size is None, returns None.
    """
    assert loader_type in ['RPIDatasetProteinRand', 'RPIDatasetRNARand', 'RPIDataset',
                           ], 'Invalid loader_type specified.'
    if seed:
        set_seed(seed)
    
    rna_embeddings, protein_embeddings = RPIDataset.pre_load_embeddings(
        rna_embeddings_path,
        protein_embeddings_path
    )

    if loader_type == 'RPIDatasetProteinRand':
        dataset = RPIDatasetProteinRand(
            rna_embeddings,
            protein_embeddings,
            dataset_path,
        )
    elif loader_type == 'RPIDatasetRNARand':
        dataset = RPIDatasetRNARand(
            rna_embeddings,
            protein_embeddings,
            dataset_path,
        )
    elif loader_type == 'RPIDataset':
        dataset = RPIDataset(
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