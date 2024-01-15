import argparse
import random
import os

import torch

import pandas as pd
import numpy as np

from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split


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


def get_dataloader(loader_type: str,
                          train_set_path: str,
                          rna_embeddings_path: str,
                          protein_embeddings_path: str,
                          val_set_size:float,
                          seed: int,
                          **kwargs
                          ):
    assert loader_type in ['PandasInMemory'], 'Invalid loader_type specified.'

    set_seed(seed)
    
    if loader_type == 'PandasInMemory':
        rna_embeddings, protein_embeddings = RNAInterActionsPandasInMemory.pre_load_embeddings(
            rna_embeddings_path,
            protein_embeddings_path
        )
        dataset = RNAInterActionsPandasInMemory(
            rna_embeddings,
            protein_embeddings,
            train_set_path,
        )

        train_set_size = 1 - val_set_size
        train_set, valid_set = random_split(dataset, [train_set_size, val_set_size])

    assert train_set is not None
    assert valid_set is not None

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


def main(args):
    
    train_dataloader, valid_dataloader = get_dataloader(loader_type=args.dataloader_type,
                                                        train_set_path=args.train_set_path,
                                                        rna_embeddings_path=args.rna_embeddings_path,
                                                        protein_embeddings_path=args.protein_embeddings_path,
                                                        val_set_size=args.val_set_size,
                                                        num_workers=args.num_workers,
                                                        batch_size=args.batch_size
                                                        )
    total_start = time()
    for idx, _ in tqdm(enumerate(iter(train_dataloader))):
        if idx == args.amount - 1:
            break
    print(f"Total time: {time() - total_start} for train_dataloader providing batches.")

    total_start = time()
    for idx, _ in tqdm(enumerate(iter(valid_dataloader))):
        if idx == args.amount - 1:
            break
    print(f"Total time: {time() - total_start} for valid_dataloader providing batches.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataloader script for RNAProteinInterAct.")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--amount", type=int, default=20, help="Logging interval")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--rna_embeddings_path", type=str, default="data/embeddings/rna_embeddings.npy", help="Path to all RNA embeddings")
    parser.add_argument("--protein_embeddings_path", type=str, default="data/embeddings/protein_embeddings.npy", help="Path to all protein embeddings")
    parser.add_argument("--train_set_path", type=str, default="data/interactions/train_set.parquet", help="Path to the train set file")
    parser.add_argument("--val_set_size", type=float, default=0.15, help="Size of the validation set in percent")
    parser.add_argument("--dataloader_type", type=str, default=None, help="Type of dataloader")

    args = parser.parse_args()

    
    main(args)