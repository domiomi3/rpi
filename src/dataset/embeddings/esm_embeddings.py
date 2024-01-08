import argparse
import sys
import os
import psutil

import esm
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from time import time
from statistics import mean

src_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_dir))
from utils import divide_dataframe


def create_embeddings(save_dir, protein_path, enable_cuda, repr_layer, max_task_id, task_id):
    """
    Create and save embeddings for proteins using the pretrained ESM-2 model.

    Args:
    - save_dir (str): path to the directory where the embeddings will be saved
    - protein_path (str): path to the parquet file containing protein data
    - enable_cuda (bool): whether to use CUDA
    - repr_layer (int): layer index from which to extract representation in the ESM-2 model
    - max_task_id (int): maximum task ID (when splitting data across multiple tasks)
    - task_id (int): current task ID
    
    Returns:
    - None
    """
    print(f"Running with task id {task_id} and max task id {max_task_id}")

    proteins_df = pd.read_parquet(protein_path, engine='pyarrow')
    
    # Split data across multiple tasks
    data_batch = divide_dataframe(proteins_df, max_task_id, task_id)
    print(f"Creating embeddings for {len(data_batch)} proteins out of {proteins_df.shape[0]} proteins from dataset {protein_path}")

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    
    print("Model loaded.")
    print("Start to create embeddings.")
    timings = []

    if enable_cuda:
        model.cuda()

    for idx, protein in enumerate(data_batch):
        
        print(f"Processing protein {idx}/{len(data_batch)}")
        if enable_cuda:
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mem = round(free_mem / 1024 / 1024 / 1024, 2)
            total_mem = round(total_mem / 1024 / 1024 / 1024, 2)
            used_mem = round(total_mem - free_mem, 2)
            print("GPU memory usage:")
            print(free_mem, total_mem, used_mem
            )
        
        start = time()

        protein_sequence = [(protein['Sequence_2_ID'], protein['Sequence_2'].upper())]

        batch_labels, _, batch_tokens = batch_converter(protein_sequence)
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
            np.save(f"{save_dir}/{protein['Sequence_2_ID']}",
                    token_representations[i, 1: tokens_len - 1].float())

        del batch_tokens
        del token_representations
        del results
        del batch_labels
        del batch_lens
        del protein_sequence

        timings.append(time() - start)

    print(f"Average time per protein embedding extraction: {round(mean(timings), 4)}")


def merge_embeddings(esm_emb_dir, emb_dir):
    """
    Merge embeddings for all proteins into a single array.

    Args:
    - esm_emb_dir (str): path to the directory containing the embeddings for each protein (.npy)
    - emb_dir (str): path to the directory where the merged embeddings will be saved

    Returns:
    - None
    """

    embeddings = []
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    paths = [(int(embed[:-4]), os.path.join(esm_emb_dir, embed)) for embed in os.listdir(esm_emb_dir) if embed.endswith('.npy')]
    paths = sorted(paths, key = lambda x: x[0])[:]

    for _, embedding_path in tqdm(paths):
        emb_protein = np.load(embedding_path)
        padded_emb_protein = np.zeros((1024, 640))
        padded_emb_protein[:emb_protein.shape[0], :] = emb_protein
        embeddings.append(padded_emb_protein)
    
    embeddings = np.stack(embeddings, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")
    # TODO: check embedding sizes
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print("Storing embeddings.")
    all_embeddings_path = os.path.join(emb_dir, "protein_embeddings.npy")
    np.save(all_embeddings_path, embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--working_dir', type=str, default='/work/dlclarge1/matusd-rpi/RPI', help='Working directory path.')
    parser.add_argument('--emb_dir', type=str, default="data/embeddings", help='Embeddings directory path.')
    parser.add_argument('--unique_proteins', type=str, default='unique_proteins.parquet', help='Unique proteins filename.')
    parser.add_argument('--enable_cuda', type=bool, default=False, help='Enable or disable CUDA')
    parser.add_argument('--repr_layer', type=int, default=30, help='Representation layer')
    parser.add_argument('--max_task_id', type=int, default=20, help='Maximum task ID')
    parser.add_argument('--task_id', type=int, default=1, help='Task ID')

    args = parser.parse_args()

    working_dir = args.working_dir
    emb_dir = args.emb_dir
    unique_proteins = args.unique_proteins
    enable_cuda = args.enable_cuda
    repr_layer = args.repr_layer
    max_task_id = args.max_task_id
    task_id = args.task_id

    os.chdir(working_dir)

    # Curate paths
    protein_path = os.path.join(emb_dir, unique_proteins)
    esm_emb_dir = os.path.join(emb_dir, "esm")
    if os.path.exists(esm_emb_dir) is False:
        os.makedirs(esm_emb_dir)

    # create_embeddings(esm_emb_dir, protein_path, enable_cuda, repr_layer, max_task_id, task_id)
    merge_embeddings(esm_emb_dir, emb_dir)