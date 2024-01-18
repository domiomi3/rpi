import argparse
import sys
import os
import psutil
import torch
import fm
import esm

import pandas as pd
import numpy as np

from tqdm import tqdm
from time import time
from statistics import mean
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))
from utils import divide_dataframe


def create_embeddings(emb_dir, data_path, model_type, enable_cuda, repr_layer, max_task_id, task_id):
    """
    Create and save embeddings for sequences using a specified model (RNA-FM or ESM-2).

    Args:
    - emb_dir (str): Directory to save the embeddings.
    - data_path (str): Path to the data file (RNA or protein data).
    - model_type (str): Type of model ('rna_fm' or 'esm2').
    - enable_cuda (bool): Whether to use CUDA.
    - repr_layer (int): Representation layer index.
    - max_task_id (int): Maximum task ID for data splitting.
    - task_id (int): Current task ID.

    Returns:
    - None
    """
    print(f"Running with task id {task_id} and max task id {max_task_id}")
    df = pd.read_parquet(data_path, engine='pyarrow')

    # Split data across multiple tasks
    data_batch = divide_dataframe(df, max_task_id, task_id)
    if len(data_batch) == 0:
        print("No data to process.")
        return
    
    print(f"Creating embeddings for {len(data_batch)} sequences out of {df.shape[0]}")

    # Load the specified model
    if model_type == 'rna_fm':
        model, alphabet = fm.pretrained.rna_fm_t12()
        idx = '1'
    elif model_type == 'esm2':
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        idx = '2'
    else:
        raise ValueError("Invalid model type. Choose 'rna_fm' or 'esm2'.")

     # Ensure the directory exists
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disables dropout for deterministic results

    if enable_cuda:
        model.cuda()

    timings = []

    for _, row in tqdm(enumerate(data_batch), total=len(data_batch), desc="Embedding sequences"):
        start = time()

        embedding_id = row[f'Sequence_{idx}_emb_ID']
        sequence = row[f'Sequence_{idx}'].upper()

        # Process sequence
        _, _, batch_tokens = batch_converter([(embedding_id, sequence)])
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
            np.save(f"{emb_dir}/{embedding_id}",
                    token_representations[i, 1: tokens_len - 1].float())
        
        del batch_tokens, token_representations, results, batch_lens

        if enable_cuda:
            torch.cuda.empty_cache()

        timings.append(time() - start)

    print(f"Average time per sequence embedding extraction: {mean(timings):.4f}")


def merge_embeddings(emb_dir, model_type):
    """
    Merge embeddings for all sequences into a single array.

    Args:
    - emb_dir (str): Path to the directory containing the embeddings.
    - model_type (str): Type of model ('rna_fm' or 'esm2').

    Returns:
    - None
    """
    sequence_type = 'rna' if model_type == 'rna_fm' else 'protein'

    # Get all .npy files from the directory
    file_paths = [os.path.join(emb_dir, f) for f in os.listdir(emb_dir) if f.endswith('.npy')]

    embeddings = []

    # Pad embeddings to the same length
    for embedding_path in tqdm(file_paths, total=len(file_paths), desc="Merging embeddings"):
        emb = np.load(embedding_path)
        padded_emb = np.zeros((1024, 640))
        padded_emb[:emb.shape[0], :] = emb
        embeddings.append(padded_emb)

    # Stack embeddings into a single array
    embeddings = np.stack(embeddings, axis=0)

    # Memory usage information (optional)
    print('RAM Used (GB):', psutil.virtual_memory()[3] / 1e9)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save the merged embeddings
    parent_dir = os.path.dirname(emb_dir)
    merged_embeddings_path = os.path.join(parent_dir, f"{sequence_type}_embeddings.npy")
    np.save(merged_embeddings_path, embeddings)
    print(f"Merged embeddings saved to {merged_embeddings_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--enable_cuda', type=bool, default=False, help='Enable or disable CUDA')
    parser.add_argument('--rna_path', type=str, default="data/annotations/unique_rna.parquet", help='Path to the RNA data (for RNA-FM model)')
    parser.add_argument('--protein_path', type=str, default="data/annotations/unique_proteins.parquet", help='Path to the protein data (for ESM-2 model)')
    parser.add_argument('--repr_layer', type=int, help='Representation layer to extract embeddings from. Set to 30 for ESM-2, 12 for RNA-FM')
    parser.add_argument('--max_task_id', type=int, default=20, help='Maximum task ID')
    parser.add_argument('--task_id', type=int, default=1, help='Task ID')
    parser.add_argument('--working_dir', type=str, default='/work/dlclarge1/matusd-rpi/RPI', help='Working directory path.')
    parser.add_argument('--emb_dir', type=str, default="data/embeddings", help='Directory to save the results')
    parser.add_argument('--model_type', type=str, choices=['rna_fm', 'esm2'], required=True, help='Type of model to use (rna_fm or esm2)')
    
    args = parser.parse_args()

    os.chdir(args.working_dir)

    if args.model_type == 'rna_fm':
        args.repr_layer = 12
        rna_fm_dir = os.path.join(args.emb_dir, "rna_fm")

        create_embeddings(rna_fm_dir, args.rna_path, args.model_type, args.enable_cuda, args.repr_layer, args.max_task_id, args.task_id)
        merge_embeddings(rna_fm_dir, args.model_type)

    elif args.model_type == 'esm2':
        args.repr_layer = 30
        esm_dir = os.path.join(args.emb_dir, "esm")

        create_embeddings(esm_dir, args.protein_path, args.model_type, args.enable_cuda, args.repr_layer, args.max_task_id, args.task_id)
        merge_embeddings(esm_dir, args.model_type)

    else:
        raise ValueError("Invalid model type. Choose 'rna_fm' or 'esm2'.")
