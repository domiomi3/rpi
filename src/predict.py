import argparse
import os
import torch

import pandas as pd

from lightning import Trainer
from tqdm import tqdm

from model import ModelWrapper
from dataloader import get_dataloader

def main(args):

    test_results_dir = os.path.join(os.getcwd(), 'test_results')
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
    
    model = ModelWrapper.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        map_location=torch.device('cpu') if args.device == 'cpu' else torch.device('cuda'),
        cpr=False,
    )
    
    model.eval()
    
    test_dataloader = get_dataloader(
        loader_type=args.loader_type,
        dataset_path=args.test_set_path,
        rna_embeddings_path=args.rna_embeddings_path,
        protein_embeddings_path=args.protein_embeddings_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    
    model = model.to(args.device)  # Ensure the model is on the correct device
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = [item.to(args.device) for item in batch]
            outputs = model.predict_step(batch, None)
            predictions.extend(outputs['logits'].cpu().numpy())
            labels.extend(outputs['labels'].cpu().numpy())

    model_name = args.checkpoint_path.split('/')[-1].split('.ckpt')[0]
    test_name = args.test_set_path.split('/')[-1].split('.parquet')[0]

    df = pd.DataFrame({'logits': predictions, 'labels': labels})
    df.to_parquet(f'results_{model_name}_{test_name}.parquet')
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Command line options for the script")
    
    parser.add_argument("--loader_type", type=str, default="RPIDataset", help="Type of dataloader")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--rna_embeddings_path", type=str, default="data/embeddings/rna_embeddings.npy", help="Path to all RNA embeddings")
    parser.add_argument("--protein_embeddings_path", type=str, default="data/embeddings/protein_embeddings.npy", help="Path to all protein embeddings")
    parser.add_argument("--test_set_path", type=str, default="data/interactions/test_set.parquet", help="Path to the test set file")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model.ckpt", help="Path to model's checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")

    args = parser.parse_args()

    main(args)
