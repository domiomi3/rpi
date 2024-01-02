from __future__ import annotations

import numpy as np
import torch.utils.data
import torch
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
import click
from torch.utils.data import random_split

"""
Contains several dataloaders:
    - for ablation cluster_experiments
    - in memory and reading from disk
"""


class RNAInterActionsPandas(torch.utils.data.Dataset):
    """
    Loads Pandas Dataframe and reads embeddings from storage when needed.
    Using this class on cluster is a bottleneck since the embeddings have to be loaded over network connection.
    """
    def __init__(self,
                 rna_embeddings_path: str,
                 protein_embeddings_path: str,
                 db_file: str,
                 ):
        self.db = pd.read_parquet(db_file, engine='pyarrow')
        self.db = self.db.assign(row_number=range(len(self.db)))
        self.protein_embeddings_path = protein_embeddings_path
        self.rna_embeddings_path = rna_embeddings_path

        self.length = self.db.shape[0]

    def __len__(self) -> int:
        """length special method"""
        # return self.n_users()
        return self.length

    def __getitem__(self, index):
        result_df = self.db[self.db['row_number'] == index][
            ['Sequence_1_ID_Unique', 'Sequence_2_ID_Unique', 'Sequence_1_shuffle', 'Sequence_2_shuffle', 'RNAInterID']]
        assert result_df.shape[0] == 1
        seq_1_ID_Unique, seq_2_ID_Unique, seq_1_shuffle, seq_2_shuffle, rna_inter_id = result_df.values.tolist()[0]

        # open embeddings
        seq_1_embed_path = os.path.join(self.rna_embeddings_path, f"{seq_1_ID_Unique}.npy")
        seq_2_embed_path = os.path.join(self.protein_embeddings_path, f"{seq_2_ID_Unique}.npy")
        interacts = float(not (seq_1_shuffle or seq_2_shuffle))
        seq_1_embed, seq_2_embed = _load_and_pad_embeddings(seq_1_embed_path, seq_2_embed_path)
        return seq_1_embed, seq_2_embed, interacts, rna_inter_id


class RNAInterActionsPandasInMemory(torch.utils.data.Dataset):
    """
    Loads Pandas Dataframe and reads embeddings from storage when needed.
    Recommended for using on cluster. Be aware that its usage needs a lot of system memory (RAM) since all embeddings
    have to be preloaded into memory.
    """
    def __init__(self,
                 rna_embeddings: np.array,
                 protein_embeddings: np.array,
                 db_file: str
                 ):
        self.db = pd.read_parquet(db_file, engine='pyarrow')
        self.db = self.db.assign(row_number=range(len(self.db)))

        self.protein_embeddings = protein_embeddings

        self.rna_embeddings = rna_embeddings

        self.length = self.db.shape[0]

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
            rna_embeddings_path: str = "dataset/rna_embeddings.npy",
            protein_embeddings_path: str = "dataset/protein_embeddings.npy", ):
        return (RNAInterActionsPandasInMemory.pre_load_rna_embeddings(rna_embeddings_path),
                RNAInterActionsPandasInMemory.pre_load_protein_embeddings(protein_embeddings_path))

    def __len__(self) -> int:
        """length special method"""
        # return self.n_users()
        return self.length

    def __getitem__(self, index):
        result_df = self.db[self.db['row_number'] == index][
            ['Sequence_1_ID_Unique', 'Sequence_2_ID_Unique', 'Sequence_1_shuffle', 'Sequence_2_shuffle', 'row_number']]
        assert result_df.shape[0] == 1
        seq_1_ID_Unique, seq_2_ID_Unique, seq_1_shuffle, seq_2_shuffle, row_number = result_df.values.tolist()[0]
        # define interactions
        interacts = float(not (seq_1_shuffle or seq_2_shuffle))
        padded_seq_1_embed = self.rna_embeddings[seq_1_ID_Unique]
        padded_seq_2_embed = self.protein_embeddings[seq_2_ID_Unique]
        # print(f"If interaction: {interacts}")
        # print(f"Sequence 1 shuffle: {seq_1_shuffle}")
        # print(f"Sequence 2 shuffle: {seq_2_shuffle}")
        return padded_seq_1_embed, padded_seq_2_embed, interacts, row_number


class RNAInterActionsPIMRNAZero(RNAInterActionsPandasInMemory):
    """
    Data Loader Class for ablation experiment.
    PIM = PandasInMemory
    Replaces the RNA input with a zero tensor.
    """
    def __getitem__(self, index):
        padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id = super().__getitem__(index)
        # Replace RNA embedding with zeros
        padded_seq_1_embed = torch.zeros(padded_seq_1_embed.shape)
        return padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id


class RNAInterActionsPIMProteinZero(RNAInterActionsPandasInMemory):
    """
        Data Loader Class for ablation experiment.
        PIM = PandasInMemory
        Replaces the Protein input with a zero tensor.
        """
    def __getitem__(self, index):
        padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id = super().__getitem__(index)
        # Replace Protein embedding with zeros
        padded_seq_2_embed = torch.zeros(padded_seq_2_embed.shape)
        return padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id


class RNAInterActionsPIMAllZero(RNAInterActionsPandasInMemory):
    """
        Data Loader Class for ablation experiment.
        PIM = PandasInMemory
        Replaces both inputs with a zero tensor.
        """
    def __getitem__(self, index):
        padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id = super().__getitem__(index)
        # Replace RNA embedding with zeros
        padded_seq_1_embed = torch.zeros(padded_seq_1_embed.shape)
        # Replace Protein embedding with zeros
        padded_seq_2_embed = torch.zeros(padded_seq_2_embed.shape)
        return padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id


class RNAInterActionsPIMAllRNA(RNAInterActionsPandasInMemory):
    """
    Data Loader Class for ablation experiment.
    PIM = PandasInMemory
    Replaces the protein input with RNA input (2x identical RNA input).
    """
    def __getitem__(self, index):
        padded_seq_1_embed, _, interacts, rna_inter_id = super().__getitem__(index)
        return padded_seq_1_embed, padded_seq_1_embed, interacts, rna_inter_id


class RNAInterActionsPIMAllProtein(RNAInterActionsPandasInMemory):
    """
    Data Loader Class for ablation experiment.
    PIM = PandasInMemory
    Replaces the RNA input with protein input (2x identical protein input).
    """
    def __getitem__(self, index):
        _, padded_seq_2_embed, interacts, rna_inter_id = super().__getitem__(index)
        return padded_seq_2_embed, padded_seq_2_embed, interacts, rna_inter_id


class RNAInterActionsPIMRNARandom(RNAInterActionsPandasInMemory):
    """
    Data Loader Class for ablation experiment.
    PIM = PandasInMemory
    Replaces the RNA input with a random tensor.
    However, keeps the padded zero values and generates random values within the range (min, max) of the original
    tensor.
    """
    def __getitem__(self, index):
        padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id = super().__getitem__(index)
        # replace seq by random
        non_zero_indices = padded_seq_1_embed.nonzero()

        random_values = (padded_seq_1_embed.min() - padded_seq_1_embed.max()) * torch.rand(
            non_zero_indices[0].shape[0]) + padded_seq_1_embed.max()
        random_seq_1 = padded_seq_1_embed.copy()
        random_seq_1[non_zero_indices] = random_values
        return random_seq_1, padded_seq_2_embed, interacts, rna_inter_id


class RNAInterActionsPIMProteinRandom(RNAInterActionsPandasInMemory):
    """
        Data Loader Class for ablation experiment.
        PIM = PandasInMemory
        Replaces the protein input with a random tensor.
        However, keeps the padded zero values and generates random values within the range (min, max) of the original
        tensor.
        """
    def __getitem__(self, index):
        padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id = super().__getitem__(index)
        # replace seq by random
        non_zero_indices = padded_seq_2_embed.nonzero()

        random_values = (padded_seq_2_embed.min() - padded_seq_2_embed.max()) * torch.rand(
            non_zero_indices[0].shape[0]) + padded_seq_2_embed.max()
        random_seq_2 = padded_seq_2_embed.copy()
        random_seq_2[non_zero_indices] = random_values
        return padded_seq_1_embed, random_seq_2, interacts, rna_inter_id


class RNAInterActionsPIMAllRandom(RNAInterActionsPandasInMemory):
    """
        Data Loader Class for ablation experiment.
        PIM = PandasInMemory
        Replaces both with random tensors.
        However, keeps the padded zero values and generates random values within the range (min, max) of the original
        tensor.
        """
    def __getitem__(self, index):
        padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id = super().__getitem__(index)
        # replace seq by random
        non_zero_indices_1 = padded_seq_1_embed.nonzero()
        random_values_1 = (padded_seq_1_embed.min() - padded_seq_1_embed.max()) * torch.rand(
            non_zero_indices_1[0].shape[0]) + padded_seq_1_embed.max()
        random_seq_1 = padded_seq_1_embed.copy()
        random_seq_1[non_zero_indices_1] = random_values_1

        non_zero_indices_2 = padded_seq_2_embed.nonzero()
        random_values_2 = (padded_seq_2_embed.min() - padded_seq_2_embed.max()) * torch.rand(
            non_zero_indices_2[0].shape[0]) + padded_seq_2_embed.max()
        random_seq_2 = padded_seq_2_embed.copy()
        random_seq_2[non_zero_indices_2] = random_values_2
        return random_seq_1, random_seq_2, interacts, rna_inter_id



def _load_and_pad_embeddings(seq_1_embed_path: str, seq_2_embed_path: str):
    seq_1_embed = np.load(seq_1_embed_path)
    seq_2_embed = np.load(seq_2_embed_path)
    # define interactions
    # pad embeddings with 0 :)

    padded_seq_1_embed = None

    # in case that RNA input embedding has
    if seq_1_embed.ndim == 2:
        padded_seq_1_embed = np.zeros((150, 640))
        padded_seq_1_embed[:seq_1_embed.shape[0], :] = seq_1_embed

    if seq_1_embed.ndim == 3:
        padded_seq_1_embed = np.zeros((150, 150, 256))
        padded_seq_1_embed[:seq_1_embed.shape[0], :seq_1_embed.shape[1], :] = seq_1_embed

    assert padded_seq_1_embed is not None
    padded_seq_2_embed = np.zeros((1024, 640))
    padded_seq_2_embed[:seq_2_embed.shape[0], :] = seq_2_embed
    return padded_seq_1_embed, padded_seq_2_embed


def get_random_dataloader(loader_type: str,
                          db_file_train: str,
                          db_file_valid: str,
                          rna_embeddings_path: str,
                          protein_embeddings_path: str,
                          **kwargs
                          ):
    assert loader_type in ['SQLite', 'Pandas', 'PandasInMemory',
                           ], 'Invalid loader_type specified.'
    if loader_type == 'SQLite':
        raise NotImplementedError()
    train_set = None
    valid_set = None
    if loader_type == 'SQLite':
        raise NotImplementedError()
    elif loader_type == 'Pandas':
        dataset = RNAInterActionsPandas(
            rna_embeddings_path,
            protein_embeddings_path,
            db_file_train,
        )
        train_set, valid_set = random_split(dataset, [0.85, 0.15])

    elif loader_type == 'PandasInMemory':
        rna_embeddings, protein_embeddings = RNAInterActionsPandasInMemory.pre_load_embeddings(
            rna_embeddings_path,
            protein_embeddings_path
        )
        if loader_type == 'PandasInMemory':
            dataset = RNAInterActionsPandasInMemory(
                rna_embeddings,
                protein_embeddings,
                db_file_train,
            )
            train_set, valid_set = random_split(dataset, [0.85, 0.15])

    assert train_set is not None
    assert valid_set is not None
    return DataLoader(train_set, shuffle=True, **kwargs), DataLoader(valid_set, shuffle=False, **kwargs)


def get_dataloader(loader_type: str,
                   db_file_train: str,
                   db_file_valid: str,
                   rna_embeddings_path: str,
                   protein_embeddings_path: str,
                   **kwargs
                   ):
    assert loader_type in ['SQLite', 'Pandas', 'PandasInMemory',
                           "PIMAllRandom", "PIMRNARandom", "PIMProteinRandom",
                           "PIMAllZero", "PIMRNAZero", "PIMProteinZero",
                           "PIMAllProtein", "PIMAllRNA"], 'Invalid loader_type specified.'
    if loader_type == 'SQLite':
        raise NotImplementedError()
        # assert dataset in ['train', 'valid', 'test'], 'Invalid dataset specified.'
    train_set = None
    valid_set = None
    if loader_type == 'SQLite':
        raise NotImplementedError()
    elif loader_type == 'Pandas':
        train_set = RNAInterActionsPandas(
            rna_embeddings_path,
            protein_embeddings_path,
            db_file_train,
        )
        valid_set = RNAInterActionsPandas(
            rna_embeddings_path,
            protein_embeddings_path,
            db_file_valid,
        )
    elif loader_type == 'PandasInMemory' or loader_type.startswith("PIM"):
        rna_embeddings, protein_embeddings = RNAInterActionsPandasInMemory.pre_load_embeddings(
            rna_embeddings_path,
            protein_embeddings_path
        )
        if loader_type == 'PandasInMemory':
            train_set = RNAInterActionsPandasInMemory(
                rna_embeddings,
                protein_embeddings,
                db_file_train,
            )
        elif loader_type == 'PIMAllRandom':
            train_set = RNAInterActionsPIMAllRandom(
                rna_embeddings,
                protein_embeddings,
                db_file_train
            )
        elif loader_type == 'PIMRNARandom':
            train_set = RNAInterActionsPIMRNARandom(
                rna_embeddings,
                protein_embeddings,
                db_file_train
            )
        elif loader_type == 'PIMProteinRandom':
            train_set = RNAInterActionsPIMProteinRandom(
                rna_embeddings,
                protein_embeddings,
                db_file_train
            )
        elif loader_type == 'PIMAllZero':
            train_set = RNAInterActionsPIMAllZero(
                rna_embeddings,
                protein_embeddings,
                db_file_train
            )
        elif loader_type == 'PIMRNAZero':
            train_set = RNAInterActionsPIMRNAZero(
                rna_embeddings,
                protein_embeddings,
                db_file_train
            )
        elif loader_type == 'PIMProteinZero':
            train_set = RNAInterActionsPIMProteinZero(
                rna_embeddings,
                protein_embeddings,
                db_file_train
            )
        elif loader_type == 'PIMAllRNA':
            train_set = RNAInterActionsPIMAllRNA(
                rna_embeddings,
                protein_embeddings,
                db_file_train
            )
        elif loader_type == 'PIMAllProtein':
            train_set = RNAInterActionsPIMAllProtein(
                rna_embeddings,
                protein_embeddings,
                db_file_train
            )

        valid_set = RNAInterActionsPandasInMemory(
            rna_embeddings,
            protein_embeddings,
            db_file_valid,
        )
    assert train_set is not None
    assert valid_set is not None
    return DataLoader(train_set, shuffle=True, **kwargs), DataLoader(valid_set, shuffle=False, **kwargs)


@click.command()
@click.option("--batch-size", default=64)
@click.option("--amount", default=20)
@click.option("--num-workers", default=1)
@click.option("--rna-embeddings-path", default="dataset/rna_embeddings.npy")
@click.option("--protein-embeddings-path", default="dataset/protein_embeddings.npy")
@click.option("--db-file-train", default="dataset/final_train_set.parquet")
@click.option("--db-file-valid", default="dataset/final_valid_set.parquet")
@click.option("--dataloader-type", default=None, type=str)
def main(batch_size: int,
         amount: int,
         num_workers: int,
         rna_embeddings_path: str,
         protein_embeddings_path: str,
         db_file_train: str,
         db_file_valid: str,
         dataloader_type: str):
    train_dataloader, valid_dataloader = get_dataloader(dataloader_type,
                                                        db_file_train,
                                                        db_file_valid,
                                                        rna_embeddings_path,
                                                        protein_embeddings_path,
                                                        num_workers=num_workers,
                                                        batch_size=batch_size
                                                        )
    total_start = time()
    for idx, x in tqdm(enumerate(iter(train_dataloader))):
        if idx == amount - 1:
            break
    print(f"Total time: {time() - total_start} for train_dataloader providing batches.")

    total_start = time()
    for idx, x in tqdm(enumerate(iter(valid_dataloader))):
        if idx == amount - 1:
            break
    print(f"Total time: {time() - total_start} for valid_dataloader providing batches.")


if __name__ == '__main__':
    main()
