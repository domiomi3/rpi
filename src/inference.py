import os
import torch
import sys
import warnings
import pandas as pd
import numpy as np
import logging

from pathlib import Path

from model import ModelWrapper

src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))
from dataset.embeddings.esm_rna_fm import create_embeddings


class Config:
    def __init__(self, **kwargs):
        self.device = "cpu"
        self.batch_size = 8
        self.num_workers = 8
        self.checkpoint_path = True
        self.emb_dir = "data/inference"
        self.enable_cuda = False
        self.max_task_id = 1
        self.task_id = 1
        self.emb_id = 1
        self.checkpoint_path = "checkpoints/epoch=0-step=100-default-seed=0-v6.ckpt"
        
        self.__dict__.update(kwargs)


def main():

    # Ignore UserWarnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    # User input
    while True:
        protein_seq = input("Enter the protein sequence: ")
        is_valid, message = check_protein(protein_seq)
        if is_valid:
            break
        else:
            print(f"Invalid protein sequence. {message}")

    while True:
        rna_seq = input("Enter the RNA sequence: ")
        is_valid, message = check_rna(rna_seq)
        if is_valid:
            break
        else:
            print(f"Invalid RNA sequence. {message}")

    config = Config()

    if not os.path.exists(config.emb_dir):
        os.makedirs(config.emb_dir)

    # Create a dataframe for each of the sequences writing the sequence and the id
    protein_df = pd.DataFrame(
        {'Sequence_2': [protein_seq], 'Sequence_2_emb_ID': [config.emb_id]}
    )
    rna_df = pd.DataFrame(
        {'Sequence_1': [rna_seq], 'Sequence_1_emb_ID': [config.emb_id]}
    )
    
    # Save the dataframe as a parquet file
    protein_path = os.path.join(config.emb_dir, 'protein.parquet')
    protein_df.to_parquet(protein_path)

    rna_path = os.path.join(config.emb_dir, 'rna.parquet')
    rna_df.to_parquet(rna_path)

    # Generate embedding for protein
    model = "esm2"
    repr_layer = 30
    esm_dir = os.path.join(config.emb_dir, 'esm')
    
    print("Generating embedding for protein...")

    create_embeddings(
        emb_dir=esm_dir,
        data_path=protein_path,
        model_type=model, 
        enable_cuda=config.enable_cuda, 
        repr_layer=repr_layer,
        max_task_id=config.max_task_id,
        task_id=config.task_id,
        if_inference=True
    )

    protein_emb = np.load(os.path.join(esm_dir, f'{config.emb_id}.npy'))
    # Pad the embedding to the same size as the RNA embedding
    padded_protein_emb = np.zeros((1024, 640))
    padded_protein_emb[:protein_emb.shape[0], :] = protein_emb
    protein_emb_tensor = torch.tensor(padded_protein_emb, dtype=torch.float32)
    protein_emb_tensor = torch.unsqueeze(protein_emb_tensor, 0) # Add batch dimension


    # Generate embedding for RNA
    model = "rna_fm"
    repr_layer = 12
    rna_fm_dir = os.path.join(config.emb_dir, 'rna_fm')

    print("Generating embedding for RNA...")

    create_embeddings(
        emb_dir=rna_fm_dir,
        data_path=rna_path,
        model_type=model, 
        enable_cuda=config.enable_cuda, 
        repr_layer=repr_layer,
        max_task_id=config.max_task_id,
        task_id=config.task_id,
        if_inference=True
    )

    rna_emb = np.load(os.path.join(rna_fm_dir, f'{config.emb_id}.npy'))
    
    # Pad the embedding to the same size as the protein embedding
    padded_rna_emb = np.zeros((1024, 640))
    padded_rna_emb[:rna_emb.shape[0], :] = rna_emb
    rna_emb_tensor = torch.tensor(padded_rna_emb, dtype=torch.float32)
    rna_emb_tensor = torch.unsqueeze(rna_emb_tensor, 0)  # Add batch dimension

    # Load the model
    model = ModelWrapper.load_from_checkpoint(
        checkpoint_path=config.checkpoint_path,
        map_location=torch.device('cpu') if config.device == 'cpu' else torch.device('cuda')
        )

    # Perform inference
    with torch.no_grad():
        prediction = model(
            rna_embed=rna_emb_tensor,
            protein_embed=protein_emb_tensor
        )

    prediction = prediction.item()
    if prediction > 0.5:
        interaction = "positive"
    else:
        interaction = "negative"
    
    print(f"-----{interaction.upper()} INTERACTION-----")


def check_rna(seq: str):
    max_length = 1024
    valid_letters = {'A', 'C', 'G', 'U'}
    unique_letters = set(seq.upper())
    invalid_letters = unique_letters.difference(valid_letters)
    if len(invalid_letters) > 0:
        return False, f"Invalid characters found: {', '.join(invalid_letters)}"
    elif len(seq) > max_length:
        return False, f"Sequence length should not exceed {max_length} characters."
    else:
        return True, ""


def check_protein(seq: str):
    max_length = 1024
    valid_letters = {'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'}
    unique_letters = set(seq.upper())
    invalid_letters = unique_letters.difference(valid_letters)
    if len(invalid_letters) > 0:
        return False, f"Invalid characters found: {', '.join(invalid_letters)}"
    elif len(seq) > max_length:
        return False, f"Sequence length should not exceed {max_length} characters."
    else:
        return True, ""


if __name__ == '__main__':
    main()
