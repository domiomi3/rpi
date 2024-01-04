import pandas as pd

from pathlib import Path

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def df2fasta(df: pd.DataFrame, fasta_path: Path, idx: str):
    """
    The function constructs a SeqRecord object for each sequence in the DataFrame
    and writes all SeqRecords to the specified FASTA file.

    Args: 
    - df (pd.DataFrame): DataFrame with sequences.
    - fasta_path (Path): Path to the FASTA file.
    - idx (str): Index of the "Sequence_X" column indicating typo of sequence 
    (1 = rna, 2 = protein).
    """
    with open(fasta_path, 'w') as handle:
        sequences = [SeqRecord(Seq(row[f'Sequence_{idx}']), id=f"{row[f'Raw_ID{idx}']}_{row[f'Sequence_{idx}_ID']}") for
                     _, row in
                     df.iterrows()]
        SeqIO.write(sequences, handle, "fasta")

def row2fasta(row: pd.Series, fasta_path: Path, idx: int):
    """
    The function constructs a SeqRecord object for a single DataFrame row
    and writes it into a specified FASTA file.
    
    Args: 
    - row (pd.core.series.Series): DataFrame with sequences.
    - fasta_path (Path): Path to the FASTA file.
    - idx (str): Index of the "Sequence_X" column indicating typo of sequence 
    (1 = rna, 2 = protein).
    """
    with open(fasta_path, 'w') as handle:
        sequence = [SeqRecord(Seq(row[f'Sequence_{idx}']), id=f"{row[f'Raw_ID{idx}']}_{row[f'Sequence_{idx}_ID']}")]
        SeqIO.write(sequence, handle, "fasta")

