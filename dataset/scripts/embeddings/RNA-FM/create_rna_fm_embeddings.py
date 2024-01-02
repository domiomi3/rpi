import torch
import fm
from time import time
from statistics import mean
import click
import numpy
import pandas as pd
import argparse

from more_itertools import divide

"""
This script helps to create RNA-FM embeddings for RNAs. 
It can be run with a array batch job SLURM cluster. 
e.g. pass SLURM_ARRAY_TASK_ID to --task-id
and SLURM_ARRAY_TASK_MAX to --max-task-id
This will distribute all RNA sequences across different jobs.
"""

# @click.command()
# @click.option('--enable-cuda', default=True)
# @click.option('--rna-path')
# @click.option('--repr-layer', default=12)
# @click.option('--max-task-id', default=0)
# @click.option('--task-id', default=0)
# @click.option('--save-dir', default="results")
# @click.option('--log-freq', default=1000)
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

def main(enable_cuda=True, rna_path="/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/results/unique_RNAs.parquet", repr_layer=12, max_task_id=0, task_id=0, save_dir="/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/RNA-FM/results", log_freq=1000):
    rna_df = pd.read_parquet(rna_path, engine='pyarrow')
    # This line divides the dataframe into (max-task-id) parts and takes the (task-id)th element.
    # data = rna_df.iloc[
    #     [list(c) for c in divide(max_task_id + 1, range(rna_df.shape[0]))][task_id]].to_dict('records')
    data = divide_dataframe(rna_df, max_task_id, task_id)
    print(f"Create embeddings for {len(data)} RNAs out of {rna_df.shape[0]} RNAs from dataset {rna_path}")
    # Load RNA-FM
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    print("Model loaded.")
    print("Start to create embeddings.")
    timings = []

    if enable_cuda:
        model.cuda()

    for idx, rna in enumerate(data):
        print(f"Processing RNA {idx}/{len(data)}")
        print(f"Sequence ID: {rna['Sequence_1_ID_Unique']}")
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_mem = round(free_mem / 1024 / 1024 / 1024, 2)
        total_mem = round(total_mem / 1024 / 1024 / 1024, 2)
        used_mem = round(total_mem - free_mem, 2)
        start = time()
        rna['Sequence_1'] = rna['Sequence_1'].upper()
        rna_sequence = [(rna['Sequence_1_ID_Unique'], rna['Sequence_1'])]

        batch_labels, batch_strs, batch_tokens = batch_converter(rna_sequence)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        if enable_cuda:
            batch_tokens = batch_tokens.to(device='cuda')
            print("Using CUDA")
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=True)
        token_representations = results["representations"][repr_layer].cpu()
        # Generate per-sequence representations
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, tokens_len in enumerate(batch_lens):
            numpy.save(f"{save_dir}/{rna['Sequence_1_ID_Unique']}",
                    token_representations[i, 1: tokens_len - 1].float())

        del batch_tokens
        del token_representations
        del results
        torch.cuda.empty_cache()
        timings.append(time() - start)
        # if idx % log_freq == 0:
        print(f"{idx}/{len(data)} embeddings done!")
        print(f"{used_mem}GB/{total_mem}GB used")

    print(f"Average time per RNA embedding extraction: {round(mean(timings), 4)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--enable_cuda', type=bool, default=True, help='Enable or disable CUDA')
    parser.add_argument('--rna_path', type=str, default='/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/results/unique_RNAs.parquet', help='Path to the protein data')
    parser.add_argument('--repr_layer', type=int, default=12, help='Representation layer')
    parser.add_argument('--max_task_id', type=int, default=20, help='Maximum task ID')
    parser.add_argument('--task_id', type=int, default=1, help='Task ID')
    parser.add_argument('--save_dir', type=str, default='/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/RNA-FM/results', help='Directory to save the results')
    parser.add_argument('--log_freq', type=int, default=1000, help='Log frequency')

    args = parser.parse_args()

    main(args.enable_cuda, args.rna_path, args.repr_layer, args.max_task_id, args.task_id, args.save_dir, args.log_freq)


