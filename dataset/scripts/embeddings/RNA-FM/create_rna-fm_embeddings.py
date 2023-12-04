import torch
import fm
from time import time
from statistics import mean
import click
import numpy
import pandas as pd
from more_itertools import divide

"""
This script helps to create RNA-FM embeddings for RNAs. 
It can be run with a array batch job SLURM cluster. 
e.g. pass SLURM_ARRAY_TASK_ID to --task-id
and SLURM_ARRAY_TASK_MAX to --max-task-id
This will distribute all RNA sequences across different jobs.
"""

@click.command()
@click.option('--enable-cuda', default=True)
@click.option('--rna-path')
@click.option('--repr-layer', default=12)
@click.option('--max-task-id', default=0)
@click.option('--task-id', default=0)
@click.option('--save-dir', default="results")
@click.option('--log-freq', default=1000)
def main(enable_cuda, rna_path, max_task_id, task_id, repr_layer, save_dir, log_freq):
    rna_df = pd.read_parquet(rna_path, engine='pyarrow')
    # This line divides the dataframe into (max-task-id) parts and takes the (task-id)th element.
    data = rna_df.iloc[
        [list(c) for c in divide(max_task_id + 1, range(rna_df.shape[0]))][task_id]].to_dict('records')
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
        if idx % log_freq == 0:
            print(f"{idx}/{len(data)} embeddings done!")
            print(f"{used_mem}GB/{total_mem}GB used")

    print(f"Average time per RNA embedding extraction: {round(mean(timings), 4)}")


if __name__ == '__main__':
    main()

