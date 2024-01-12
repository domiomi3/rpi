import argparse
import os

import numpy as np
import pandas as pd

from tqdm import tqdm


def create_embeddings(df: pd.DataFrame, enc_size, alphabet: dict, idx: str):
    """
    Create one-hot-encoding for provided sequences.

    Args:
    - df (pd.DataFrame): DataFrame containing sequences.
    - enc_size (int): Length to which the sequences are padded or truncated.
    - alphabet (dict): Mapping from characters to integer indices.
    - seq_key (str): Column name in DataFrame for the sequences.

    Returns:
    - (list, np.ndarray): List of encoded sequences and array of one-hot encoded sequences.
    """
    seqs = []
    encoded_seqs = []
    df = df.sort_values(by=[f"Sequence_{idx}_ID"])

    for _, row in tqdm(df.iterrows(), total=len(df)):
        seq = row[f"Sequence_{idx}"]
        assert isinstance(seq, str)
        for elm in seq:
            assert elm.upper() in alphabet
        enc_seq = np.array([alphabet[elm.upper()] for elm in seq])
        enc_dimension = len(alphabet)
        one_enc_seq = np.zeros((enc_size, enc_dimension), dtype=int)
        one_enc_seq[np.arange(enc_seq.size), enc_seq] = 1 
        seqs.append(enc_seq)
        encoded_seqs.append(one_enc_seq)
    encoded_seqs = np.stack(encoded_seqs)
    return seqs, encoded_seqs


def collect_alphabet(df: pd.DataFrame, seq_key: str):
    """
    Collects and prints the alphabet of sequences in a DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing sequences.
    - seq_key (str): Column name in DataFrame for the sequences.
    """
    letters = set()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        seq = row[seq_key]
        assert isinstance(seq, str)
        for elm in seq:
            letters.add(elm.upper())
    letters = sorted(list(letters))
    print({letter: idx for idx, letter in enumerate(letters)})


def main(args):
    # Create output directory
    one_hot_dir = os.path.join(args.emb_dir, "one_hot")
    if not os.path.exists(one_hot_dir):
        os.makedirs(one_hot_dir, exist_ok=True)

    # Load unique RNA sequences and create one-hot-encodings
    if args.rna_path:
        unique_RNAs = pd.read_parquet(args.rna_path)
        _, rna_embeddings = create_embeddings(unique_RNAs, args.rna_enc_size, args.rna_alphabet, '1')
        np.save(one_hot_dir, "one_hot_RNA.npy", rna_embeddings) 

    # Load unique protein sequences and create one-hot-encodings
    if args.protein_path:
        unique_proteins = pd.read_parquet(args.protein_path)
        _, protein_embeddings = create_embeddings(unique_proteins, args.protein_enc_size, args.protein_alphabet, '2')
        np.save(os.path.join(one_hot_dir, "one_hot_proteins.npy"), protein_embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create One-Hot Encodings for RNA and Protein Sequences")

    parser.add_argument('--working_dir', type=str, default="/work/dlclarge1/matusd-rpi/RPI/", help="Working dir")
    parser.add_argument('--emb_dir', type=str, default="data/embeddings/", help="Path to save RNA embeddings")
    parser.add_argument('--rna_path', type=str, default="unique_RNA.parquet", help='Path to RNA parquet file')
    parser.add_argument('--protein_path', type=str, default="unique_proteins.parquet", help='Path to Protein parquet file')
    parser.add_argument('--rna_enc_size', type=int, default=1024, help='Encoding size for RNA sequences')
    parser.add_argument('--protein_enc_size', type=int, default=1024, help='Encoding size for protein sequences')
    parser.add_argument('--rna_alphabet', type=dict, default={'C': 0, 'G': 1, 'A': 2, 'U': 3}, help='Alphabet for RNA sequences')
    parser.add_argument('--protein_alphabet', type=dict, default={'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'X': 19, 'Y': 20}, help='Alphabet for protein sequences')

    args = parser.parse_args()
    main(args)