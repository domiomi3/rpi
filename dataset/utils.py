import os

import pandas as pd
import requests as r
import concurrent.futures

import numpy as np

from random import choice
from pathlib import Path
from more_itertools import chunked
from tqdm import tqdm

from Bio import SeqIO, Entrez
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


def load_rna_inter(database: str, path="../Download_data_RP.txt"):
    # RAW_ID_1 is always the RNA Interactor
    rna_inter_df = load_rna_inter_csv(path)
    rna_inter_df = rna_inter_df[
        rna_inter_df['Raw_ID1'].str.startswith(f"{database}:", na=False)]
    rna_inter_df['RNA_DB'] = rna_inter_df['Raw_ID1'].str.split(':').str[0]
    return rna_inter_df


def load_rna_inter_csv(path: str):
    """
    Loads the RNA Inter CSV file and returns a DataFrame with specified columns.
    """
    dtypes = {
        "RNAInterID": str,
        "Interactor1.Symbol": str,
        "Category1": str,
        "Species1": str,
        "Interactor2.Symbol": str,
        "Category2": str,
        "Species2": str,
        "Raw_ID1": str,
        "Raw_ID2": str,
        "score": float,
        "strong": str,
        "weak": str,
        "predict": str,
    }
    return pd.read_csv(path, sep='\t', dtype=dtypes)


def divide_dataframe(df, max_task_id, task_id):
    """
    Divide a DataFrame into equal parts based on max_task_id and return the part specified by task_id.

    :param df: pandas DataFrame to be divided.
    :param max_task_id: The total number of parts to divide the DataFrame into.
    :param task_id: The index of the part to return (0-indexed).
    :return: The subset of the DataFrame corresponding to the task_id.
    """
    # If max_task_id is 1, return the entire DataFrame
    if max_task_id == 1:
        return df.to_dict('records')
    
    # Calculate the number of rows per task
    rows_per_task = len(df) // max_task_id

    # Calculate the start and end index for the slice
    start_idx = task_id * rows_per_task
    end_idx = start_idx + rows_per_task

    # Adjust the end index for the last task to include any remaining rows
    if task_id == max_task_id - 1:
        end_idx = len(df)

    # Return the subset of the DataFrame
    return df.iloc[start_idx:end_idx].to_dict('records')


def call_fetch_function(func_name, chunk_size, ids, retries=3) -> pd.DataFrame:
    ids_chunks = list(chunked(ids, chunk_size))
    results = []
    for idx, ids_chunk in enumerate(tqdm(ids_chunks)):
        except_counter = 0
        for _ in range(retries):
            try:
                results += func_name(ids_chunk)
                break
            except Exception as e:
                print(e)
                except_counter += 1
                continue
        if except_counter == retries:
            print(f"More than three errors occured for chunk with index {idx} (and chunk size: {chunk_size})")
    return pd.DataFrame(results)


def show_rna_inter_protein_databases():
    rna_inter_df = pd.read_csv("../Download_data_RP.txt", sep='\t')
    rna_inter_df['Protein_DB'] = rna_inter_df['Raw_ID2'].str.split(':').str[0]
    rna_inter_df['Protein_ID'] = rna_inter_df['Raw_ID2'].str.split(':').str[1]
    print(list(rna_inter_df['Protein_DB'].unique()))


def load_rna_inter_protein(database: str, path="../Download_data_RP.txt"):
    # RAW_ID_1 is always the RNA Interactor
    rna_inter_df = load_rna_inter_csv(path)
    rna_inter_df = rna_inter_df[
        rna_inter_df['Raw_ID2'].str.startswith(f"{database}:", na=False)]
    rna_inter_df['Protein_DB'] = rna_inter_df['Raw_ID1'].str.split(':').str[0]
    return rna_inter_df


def analyze_rna_inter_scores(df: pd.DataFrame):
    ranges = np.arange(0.0, 1.1, 0.1)
    scores_ranges = dict()

    scores = list(df['score'])
    scores = np.array(scores)
    for idx in range(1, len(ranges)):
        cond = (ranges[idx - 1] <= scores) & (scores < ranges[idx])
        scores_ranges[f"[{ranges[idx - 1]:.2f}, {ranges[idx]:.2f})"] = len(scores[cond])
    return scores_ranges


def get_sequences_by_id_with_interval(sequence_id: int, seq_start: int, seq_stop: int):
    Entrez.email = "gernel@informatik.uni-freiburg.de"
    for _ in range(3):
        try:
            handle = Entrez.efetch(db="nuccore", id=sequence_id, seq_start=seq_start, seq_stop=seq_stop,
                                   rettype="fasta")
            return (sequence_id, list(SeqIO.parse(handle, "fasta")))
        except Exception as e:
            print(e)
    return -1


def analyze_sequence_lens(sizes: list, df: pd.DataFrame):
    df_sizes = {}
    for seq_size in sizes:
        df_size = df[df['sequence_len'] < seq_size].shape[0]
        df_sizes[seq_size] = df_size
        print(f"There are {df_size:,} interactions with a length < {seq_size}")
    print(f"There are {df.shape[0]} entries.")
    print(f"Max sequence length: {df['sequence_len'].max()}")
    return df_sizes


def analyze_protein_sequence_lens(sizes: list, df: pd.DataFrame):
    df_sizes = {}
    for seq_size in sizes:
        df_size = df[df['protein_sequence_len'] < seq_size].shape[0]
        df_sizes[seq_size] = df_size
        print(f"There are {df_size:,} proteins with a length < {seq_size}")
    print(f"Proteins in total: {df.shape[0]}")
    print(f"Max proteins sequence length: {df['protein_sequence_len'].max()}")
    print(f"Average protein sequence length: {df['protein_sequence_len'].mean()}")
    return df_sizes


def get_protein_fasta(cID):
    base_url = "http://www.uniprot.org/uniprot/"
    current_url = base_url + cID + ".fasta"
    for _ in range(3):
        try:
            response = r.post(current_url)
            if response.ok:
                return cID, ''.join(response.text)
        except Exception as e:
            print(e)
            pass
    print("More than three errors!")
    return cID, -1


def calc_recovery_rate(rna_inter_df: pd.DataFrame, database_df: pd.DataFrame, col_name='Raw_ID1'):
    before_extraction_count = rna_inter_df[col_name].nunique()
    after_extraction_count = database_df[col_name].nunique()
    print(f"Unique Gene IDs before extraction:\t{before_extraction_count:,}")
    print(f"Unique Gene IDs after extraction:\t{after_extraction_count:,}")
    print(f"Extraction rate:\t{round(after_extraction_count / before_extraction_count * 100, 2)}%")


def remove_illegal_nucleotides(df: pd.DataFrame, nucleotides: list) -> pd.DataFrame:
    if isinstance(nucleotides, str):
        nucleotides = [nucleotides]
    for nucleotide in nucleotides:
        assert len(nucleotide) == 1, "Nucleotide should be a single character"
        df[f"has_{nucleotide}"] = df['Sequence_1'].str.contains(nucleotide)
        df = df[df[f"has_{nucleotide}"] == False]
        df = df.drop([f"has_{nucleotide}"], axis=1)
    return df


def check_sequences(df: pd.DataFrame):
    assert 'Sequence_1' in df.columns, "There are no sequences present in the dataframe"
    for idx, row in df.iterrows():
        assert isinstance(row['Sequence_1'], str), f"Element on index {idx} is not a string"
        assert len(row['Sequence_1']) > 0, f"Sequence on index {idx} is smaller than 1‚"
        for nucleotide in row['Sequence_1']:
            assert nucleotide in ['A', 'C', 'U', 'G'], f"Illegal nucleotide \"{nucleotide}\" on index {idx}"


def fetch_ncbi_rna_fasta_with_range(seqs: list):
    results = []
    assert len(seqs) > 0
    assert 'seq_id' in seqs[0]
    assert 'seq_start' in seqs[0]
    assert 'seq_end' in seqs[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit the tasks to the executor and collect the futures
        futures = [executor.submit(
            get_sequences_by_id_with_interval,
            seq['seq_id'], seq['seq_start'], seq['seq_end']) for seq in seqs]
        # Wait for the futures to complete and print the results
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            if idx % 100 == 0 and idx != 0:
                print(f"{idx}/{len(seqs)} rna sequences fetched")
                print(f"{round((idx / len(seqs) * 100), 2)}")
            results.append(future.result())
    return results


def fetch_protein_fasta(object_ids):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit the tasks to the executor and collect the futures
        futures = [executor.submit(
            get_protein_fasta,
            object_id) for object_id in object_ids]
        # Wait for the futures to complete and print the results
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            if idx % 1000 == 0 and idx != 0:
                print(f"{idx}/{len(object_ids)} proteins done")
            results.append(future.result())
    return results