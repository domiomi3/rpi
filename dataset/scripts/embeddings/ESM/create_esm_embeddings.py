import torch
import esm
from time import time
from statistics import mean
import click
import numpy
import pandas as pd
from more_itertools import divide

"""
This script helps to create ESM embeddings for proteins. 
It can be run with a array batch job SLURM cluster. 
e.g. pass SLURM_ARRAY_TASK_ID to --task-id
and SLURM_ARRAY_TASK_MAX to --max-task-id
This will distribute all protein sequences across different jobs.
Recommended to use with pre-trained model: esm2_t30_150M_UR50D ==> --repr-layer=30
(see https://github.com/facebookresearch/esm#available-models-and-datasets-) 
"""


@click.command()
@click.option('--enable-cuda', default=True)
@click.option('--model-name')
@click.option('--protein-path')
@click.option('--repr-layer', type=int)
@click.option('--max-task-id', default=0)
@click.option('--task-id', default=0)
@click.option('--save-dir', default="results")
@click.option('--log-freq', default=1000)
def main(enable_cuda, model_name, protein_path, max_task_id, task_id, repr_layer, save_dir, log_freq):
    proteins_df = pd.read_parquet(protein_path, engine='pyarrow')
    # This line divides the dataframe into (max-task-id) parts and takes the (task-id)th element.
    data = proteins_df.iloc[
        [list(c) for c in divide(max_task_id + 1, range(proteins_df.shape[0]))][task_id]].to_dict('records')
    print(f"Create embeddings for {len(data)} proteins out of {proteins_df.shape[0]} proteins from dataset {protein_path}")
    print(f"Model name: {model_name}")

    # Load ESM-2 model
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(f"models/{model_name}.pt")
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    print("Model loaded.")
    print("Start to create embeddings.")
    timings = []

    # data = [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")]
    if enable_cuda:
        model.cuda()

    for idx, protein in enumerate(data):
        if enable_cuda:
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mem = round(free_mem / 1024 / 1024 / 1024, 2)
            total_mem = round(total_mem / 1024 / 1024 / 1024, 2)
            used_mem = round(total_mem - free_mem, 2)
        start = time()
        protein['Sequence_2'] = protein['Sequence_2'].upper()
        protein_sequence = [(protein['Sequence_2_ID_Unique'], protein['Sequence_2'])]
        # protein_sequence = [protein]

        batch_labels, batch_strs, batch_tokens = batch_converter(protein_sequence)
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
            numpy.save(f"{save_dir}/{protein['Sequence_2_ID_Unique']}",
                    token_representations[i, 1: tokens_len - 1].float())

        del batch_tokens
        del token_representations
        del results
        if enable_cuda:
            torch.cuda.empty_cache()
            if idx % log_freq == 0:
                print(f"{idx}/{len(data)} embeddings done!")
                print(f"{used_mem}GB/{total_mem}GB used")
        timings.append(time() - start)

    print(f"Average time per protein embedding extraction: {round(mean(timings), 4)}")

if __name__ == '__main__':
    main()

