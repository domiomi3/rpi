from __future__ import annotations

import argparse
import os
import torch

import numpy as np
import pandas as pd

from time import time
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm


class RNAInterActionsPandas(Dataset):
    """
    Loads Pandas Dataframe and reads embeddings from storage when needed.
    Using this class on cluster is a bottleneck since the embeddings have to be loaded over network connection.
    """
    def __init__(self,
                 rna_embeddings_dir: str,
                 protein_embeddings_dir: str,
                 db_file: str,
                 ):
        self.db = pd.read_parquet(db_file, engine='pyarrow')
        # Create row number column
        self.db = self.db.assign(row_number=range(len(self.db)))
        self.protein_embeddings_dir = protein_embeddings_dir
        self.rna_embeddings_dir = rna_embeddings_dir

        self.length = self.db.shape[0]

    def __len__(self) -> int:
        """length special method"""
        # return self.n_users()
        return self.length

    def __getitem__(self, index):
        result_df = self.db[self.db['row_number'] == index][
            ['Sequence_1_ID', 'Sequence_2_ID', 'interaction', 'RNAInterID']]
        assert result_df.shape[0] == 1
        seq_1_ID, seq_2_ID, interaction, rna_inter_id = result_df.values.tolist()[0]

        # open embeddings
        seq_1_embed_path = os.path.join(self.rna_embeddings_dir, f"{seq_1_ID}.npy")
        seq_2_embed_path = os.path.join(self.protein_embeddings_dir, f"{seq_2_ID}.npy")
        interacts = float(interaction)
        seq_1_embed, seq_2_embed = _load_and_pad_embeddings(seq_1_embed_path, seq_2_embed_path)
        return seq_1_embed, seq_2_embed, interacts, rna_inter_id


class RNAInterActionsPandasInMemory(Dataset):
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
        rna_embeddings = np.load(rna_embeddings_path, allow_pickle=True)
        print(f"Loaded RNA embeddings into RAM in {round(time() - start, 2)} seconds")
        return rna_embeddings

    @staticmethod
    def pre_load_protein_embeddings(
            protein_embeddings_path: str
    ):
        print("Loading Protein embeddings...")
        start = time()
        protein_embeddings = np.load(protein_embeddings_path, allow_pickle=True)
        print(f"Loaded Protein embeddings into RAM in {round(time() - start, 2)} seconds")
        return protein_embeddings

    @staticmethod
    def pre_load_embeddings(
            rna_embeddings_path: str = "data/embeddings/rna_embeddings.npy",
            protein_embeddings_path: str = "data/embeddings/protein_embeddings.npy", ):
        return (RNAInterActionsPandasInMemory.pre_load_rna_embeddings(rna_embeddings_path),
                RNAInterActionsPandasInMemory.pre_load_protein_embeddings(protein_embeddings_path))

    def __len__(self) -> int:
        """length special method"""
        # return self.n_users()
        return self.length

    def __getitem__(self, index):
        result_df = self.db[self.db['row_number'] == index][
            ['Sequence_1_emb_ID', 'Sequence_2_emb_ID', 'interaction', 'row_number']]    
        seq_1_emb_id, seq_2_emb_id, interaction, row_number = result_df.values.tolist()[0]

        assert row_number == index

        interaction = float(interaction)

        # Extract embeddings

        padded_seq_1_embed = self.rna_embeddings[seq_1_emb_id]
        padded_seq_2_embed = self.protein_embeddings[seq_2_emb_id]

        return padded_seq_1_embed, padded_seq_2_embed, interaction, row_number


def _load_and_pad_embeddings(seq_1_embed_path: str, seq_2_embed_path: str):
    seq_1_embed = np.load(seq_1_embed_path)
    seq_2_embed = np.load(seq_2_embed_path)

    padded_seq_1_embed = np.zeros((1024, 640))
    padded_seq_1_embed[:seq_1_embed.shape[0], :] = seq_1_embed

    padded_seq_2_embed = np.zeros((1024, 640))
    padded_seq_2_embed[:seq_2_embed.shape[0], :] = seq_2_embed
    assert padded_seq_2_embed is not None
    
    return padded_seq_1_embed, padded_seq_2_embed


def get_dataloader(loader_type: str,
                          db_file_train: str,
                          rna_embeddings_path: str,
                          protein_embeddings_path: str,
                          **kwargs
                          ):
    
    torch.manual_seed(55)

    assert loader_type in ['Pandas', 'PandasInMemory',
                           ], 'Invalid loader_type specified.'
    
    if loader_type == 'Pandas':

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
        dataset = RNAInterActionsPandasInMemory(
            rna_embeddings,
            protein_embeddings,
            db_file_train,
        )
        train_set, valid_set = random_split(dataset, [0.85, 0.15])

    assert train_set is not None
    assert valid_set is not None
    return DataLoader(train_set, shuffle=True, **kwargs), DataLoader(valid_set, shuffle=False, **kwargs)



def main(args):
    train_dataloader, valid_dataloader = get_dataloader(args.dataloader_type,
                                                        args.db_file_train,
                                                        args.rna_embeddings_path,
                                                        args.protein_embeddings_path,
                                                        num_workers=args.num_workers,
                                                        batch_size=args.batch_size,
                                                        )

    total_start = time.time()
    for idx, _ in tqdm(enumerate(iter(train_dataloader))):
        if idx == args.amount - 1:
            break
    print(f"Total time: {time.time() - total_start} for train_dataloader providing batches.")

    total_start = time.time()
    for idx, _ in tqdm(enumerate(iter(valid_dataloader))):
        if idx == args.amount - 1:
            break
    print(f"Total time: {time.time() - total_start} for valid_dataloader providing batches.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--amount", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--rna_embeddings_path", type=str, default="data/embeddings/rna_embeddings.npy")
    parser.add_argument("--protein_embeddings_path", type=str, default="data/embeddings/protein_embeddings.npy")
    parser.add_argument("--db_file_train", type=str, default="/work/dlclarge1/matusd-rpi/RPI/data/interactions/train_set.parquet")
    parser.add_argument("--dataloader_type", type=str, default=None)
    

    args = parser.parse_args()
    main(args)
