import argparse
import os

import numpy as np
import pandas as pd

from tqdm import tqdm


def create_embeddings(df, enc_size, alphabet, idx):
    """
    Create one-hot-encodings for provided sequences.

    Args:
    - df (pd.DataFrame): DataFrame containing sequences.
    - enc_size (int): Length to which the sequences are padded or truncated.
    - alphabet (dict): Mapping from characters to integer indices.
    - idx (str): 2 for protein, 1 for RNA.

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


def main(idx, emb_dir, seq_path, enc_size, alphabet):
    # Load unique RNA sequences and create one-hot-encodings
    sequence_type = 'rna' if idx == '1' else 'protein'

    unique_seq = pd.read_parquet(seq_path)
    _, one_hot_embeddings = create_embeddings(unique_seq, enc_size, alphabet, idx)
    
    print(f"Embeddings shape: {one_hot_embeddings.shape}")
    
    np.save(os.path.join(emb_dir, f"one_hot_{sequence_type}.npy"), one_hot_embeddings) 
    print(f"Saved one-hot-encodings for {sequence_type} sequences to {os.path.join(emb_dir, f'one_hot_{sequence_type}.npy')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create One-Hot Encodings for RNA and Protein Sequences")

    parser.add_argument('--working_dir', type=str, default="/work/dlclarge1/matusd-rpi/RPI/", help="Working dir")
    parser.add_argument('--emb_dir', type=str, default="data/embeddings/", help="Path to save RNA embeddings")
    parser.add_argument('--rna_path', type=str, default="data/annotations/unique_rna.parquet", help='Path to RNA parquet file')
    parser.add_argument('--protein_path', type=str, default="data/annotations/unique_proteins.parquet", help='Path to Protein parquet file')
    parser.add_argument('--rna_enc_size', type=int, default=1024, help='Encoding size for RNA sequences')
    parser.add_argument('--protein_enc_size', type=int, default=1024, help='Encoding size for protein sequences')
    parser.add_argument('--rna_alphabet', type=dict, default={'C': 0, 'G': 1, 'A': 2, 'U': 3}, help='Alphabet for RNA sequences')
    parser.add_argument('--protein_alphabet', type=dict, default={'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'X': 19, 'Y': 20}, help='Alphabet for protein sequences')
    parser.add_argument('--sequence_type', type=str, choices=['rna', 'protein'], required=True, help='Sequence type to be embedded (rna or protein)')

    args = parser.parse_args()

    os.chdir(args.working_dir)

    if args.sequence_type == 'rna':
        main('1', args.emb_dir, args.rna_path, args.rna_enc_size, args.rna_alphabet)

    elif args.sequence_type == 'protein':
        main('2', args.emb_dir, args.protein_path, args.protein_enc_size, args.protein_alphabet)

    else:
        raise ValueError(f"Invalid sequence type: {args.sequence_type}. Must be either 'rna' or 'protein'.")