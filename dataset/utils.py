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


def create_negative_dataset_per_interactor(positive_dataset: pd.DataFrame, interactor_idx: int):
    """
    Creates a negative dataset for a given interactor (protein or RNA). For every positive interaction in the dataset,
    it creates a negative one with interactor changed based on cluster, family and category information.

    Args:
    - positive_dataset: DataFrame with positive interactions only
    - interactor_idx: fixed interactor that is to be paired for a negative interaction with the changing interactor,
    (1 = RNA, 2 = protein).

    Returns:
    - negative_interactions: DataFrame with negative interactions only
    """
    # Get the other interactor
    changing_interactor_idx = 1 if interactor_idx == 2 else 2
    sequence_type = 'RNA' if changing_interactor_idx == 1 else 'protein'

    negative_interactions = []

    for _, row in tqdm(positive_dataset.iterrows(), total=positive_dataset.shape[0]):

        # Get all entries that include the interactor and get cluster, family and category info about the changing interactor
        interacting_rows = positive_dataset[positive_dataset[f'Raw_ID{interactor_idx}'] == row[f'Raw_ID{interactor_idx}']]
        
        # Family information is only available for RNAs
        if changing_interactor_idx == 1:
            interacting_families = interacting_rows[f'Sequence_{changing_interactor_idx}_family'].unique()
        interacting_clusters = interacting_rows[f'Sequence_{changing_interactor_idx}_cluster'].unique()
        interacting_categories = interacting_rows[f'Category{changing_interactor_idx}'].unique()

        # Exclude all entries whose changing_interactor is fo the same category as the current row's one
        possible_interactors = positive_dataset[positive_dataset[f'Category{changing_interactor_idx}'].isin(interacting_categories)]
        if not possible_interactors.empty:
            
            # Exclude all entries whose changing_interactor is in the same cluster as the current row's one
            cl_possible_interactors = possible_interactors[~possible_interactors[f'Sequence_{changing_interactor_idx}_cluster'].isin(interacting_clusters)]
            if not cl_possible_interactors.empty:
                
                # Family info only available for RNAs
                if changing_interactor_idx == 1:

                    # Exclude all entries whose changing_interactor is in the same family as the current row's one
                    fam_possible_interactors = cl_possible_interactors[cl_possible_interactors[f'Sequence_{changing_interactor_idx}_family'].isin(interacting_families)]
                    
                    if not fam_possible_interactors.empty:                        
                        neg_interaction = create_negative_interaction(row, fam_possible_interactors, interactor_idx)
                        negative_interactions.append(neg_interaction)
                    else:
                        # print(f"Couldn't find {sequence_type} within different family. Assigning {sequence_type} based on cluster information.")
                        
                        neg_interaction = create_negative_interaction(row, cl_possible_interactors, interactor_idx)
                        negative_interactions.append(neg_interaction)
                
                # Protein search ends here
                else:
                    neg_interaction = create_negative_interaction(row, cl_possible_interactors, interactor_idx)
                    negative_interactions.append(neg_interaction)
            else:
                # print(f"Couldn't find {sequence_type} within different cluster. Assigning {sequence_type} based on category.")

                neg_interaction = create_negative_interaction(row, possible_interactors, interactor_idx)
                negative_interactions.append(neg_interaction)
        else:
            print(f"Couldn't find {sequence_type} with different category. Assigning {sequence_type} randomly.")
            possible_interactors = positive_dataset[~positive_dataset.index.isin(interacting_rows.index)]

            neg_interaction = create_negative_interaction(row, possible_interactors, interactor_idx)
            negative_interactions.append(neg_interaction)

    return pd.DataFrame(negative_interactions)


def create_negative_interaction(positive_interaction, possible_interactors, interactor_idx):
    """
    Creates a negative interaction by randomly selecting a changing interactor from the possible interactors and combining new changing interactor information
    with the interactor information from the positive interaction example.

    Args:
    - positive_interaction (pd.Series): Positive RPI entry.
    - possible_interactors (pd.DataFrame): Filtered DataFrame.
    - interactor_idx (int): Index of the fixed interactor (1 = RNA, 2 = protein).
    
    Returns:
    - negative_interaction (pd.Series): Negative RPI entry.
    """
    changing_interactor_idx = 1 if interactor_idx == 2 else 2

    negative_interaction = positive_interaction.copy()
    
    # From all possible interactors, select one at random
    random_changing_interactor_id = choice(possible_interactors[f'Raw_ID{changing_interactor_idx}'].unique())
    changing_interactor = possible_interactors[possible_interactors[f'Raw_ID{changing_interactor_idx}'] == random_changing_interactor_id].iloc[0]
   
    # Select all columns that are relevant for the changing_interactor
    interactor_columns = [
                            f'Raw_ID{changing_interactor_idx}', f'Interactor{changing_interactor_idx}.Symbol', f'Category{changing_interactor_idx}',
                            f'Species{changing_interactor_idx}', f'Sequence_{changing_interactor_idx}', f'Sequence_{changing_interactor_idx}_len',
                            f'Sequence_{changing_interactor_idx}_ID', f'Sequence_{changing_interactor_idx}_cluster',
                            f'Sequence_{changing_interactor_idx}_cluster_sim', f'Sequence_{changing_interactor_idx}_cluster_reference',
                            ]
    if changing_interactor_idx == 1:
        interactor_columns.extend(['Sequence_1_rfam_q_accession', 'Sequence_1_family',
        'Sequence_1_rfam_t_accession', 'Sequence_1_rfam_description',
        'Sequence_1_rfam_e_value'])
    
    # Copy all relevant columns from the chosen changing_interactor to the negative interaction entry
    for column in interactor_columns:
        negative_interaction[column] = changing_interactor[column]

    # Indicate it's a negative interaction 
    negative_interaction['interaction'] = False
    negative_interaction['RNAInterID'] = f'{negative_interaction["RNAInterID"]}_N'
    
    # Set remaining columns to NaN
    negative_interaction['score'] = 0
    negative_interaction['strong'] = "NaN" 
    negative_interaction['weak'] = "NaN"
    negative_interaction['predict'] = "NaN"

    return negative_interaction


def divide_dataframe(df, max_task_id, task_id):
    """
    Divide a DataFrame into equal parts based on max_task_id and return the part specified by task_id.

    :param df: pandas DataFrame to be divided.
    :param max_task_id: The total number of parts to divide the DataFrame into.
    :param task_id: The index of the part to return (0-indexed).
    :return: The subset of the DataFrame corresponding to the task_id.
    """
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

if __name__ == "__main__":
    breakpoint()