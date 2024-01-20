import argparse
import os
import torch

from lightning import Trainer

from model import ModelWrapper
from dataloader import get_dataloader


def main(args):

    test_results_dir = os.path.join(os.getcwd(), 'test_results')
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
    
    model = ModelWrapper.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        map_location=torch.device('cpu') if args.device == 'cpu' else torch.device('cuda')
    )
    
    test_dataloader, _ = get_dataloader(
        loader_type=args.loader_type,
        dataset_path=args.test_set_path,
        rna_embeddings_path=args.rna_embeddings_path,
        protein_embeddings_path=args.protein_embeddings_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    trainer = Trainer(accelerator=args.device)
    trainer.test(model, dataloaders=test_dataloader)

    model_name = args.checkpoint_path.split('/')[-1].split('.ckpt')[0]
    test_name = args.test_set_path.split('/')[-1].split('.parquet')[0]

    with open(os.path.join(test_results_dir, f'{model_name} on {test_name}.txt'), 'w') as file:
        for key, value in trainer.logged_metrics.items():
            file.write(f'{key}: {value}\n')
    
    
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
