import sys
import argparse

import torch
import lightning.pytorch.loggers

from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

src_dir = Path.cwd().parent
sys.path.append(str(src_dir))
from model import RNAProteinInterAct, RNAProteinInterActSE, ModelWrapper
from dataloader import get_dataloader


def main(args):
    
    if args.one_hot_encoding:
        model = RNAProteinInterActSE
    else:
        model = RNAProteinInterAct

    rpi_model = model( 
                        batch_first=True,
                        embed_dim=640,
                        d_model=args.d_model,
                        num_encoder_layers=args.num_encoder_layers,
                        nhead=args.n_head,
                        dim_feedforward=args.dim_feedforward,
                        key_padding_mask=args.key_padding_mask,
                        norm_first=True,
                        dropout=args.dropout
                     )
    
    lightning_module = ModelWrapper(rpi_model,
                                    lr_init=args.lr_init,
                                    weight_decay=args.weight_decay,
                                    seed=args.seed,
                                    )
    if args.compiled:
        lightning_module = torch.compile(lightning_module)
    
    # True is the default value of the logger.
    logger = True

    if args.wandb:
        logger = lightning.pytorch.loggers.WandbLogger(
            project="RNAProteinInterAct",
            name=f"{args.wandb_run_name}, seed: {args.seed}")
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints_dir,  # Custom path for saving checkpoints
        filename='{epoch}-{step}-' + f"{args.wandb_run_name}-seed={args.seed}",  # Filename format
        monitor='train_loss',  # Metric to monitor for best models
        mode='min',  # Mode for the monitored metric, 'min' for minimization
        save_last=True,  # Save the last checkpoint in addition to the best ones
    )

    trainer = Trainer(accelerator=args.accelerator,
                      devices=args.devices,
                      max_epochs=args.max_epochs,
                      logger=logger,
                      # limit_val_batches=0.0,
                      callbacks=[lr_monitor, checkpoint_callback]
                      )
    
    train_dataloader, val_dataloader = get_dataloader(  
        dataset_path=args.train_set_path,
        rna_embeddings_path=args.rna_embeddings_path,
        protein_embeddings_path=args.protein_embeddings_path,
        seed=args.seed,
        split_set_size=args.val_set_size,
        num_workers=args.num_dataloader_workers,
        batch_size=args.batch_size
    )

    trainer.fit(model=lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Command line options for the script")

    parser.add_argument("--accelerator", default='cpu', help="Type of accelerator")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--compiled", action='store_true', default=False, help="Enables Torch compiled Module")
    parser.add_argument("--wandb", action='store_true', default=False, help="Enables logging via wandb")
    parser.add_argument("--wandb_run_name", default="default", help="Name of the wandb run")
    
    parser.add_argument("--one_hot_encoding", action='store_true', default=False, help="Enables one-hot encoding")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--d_model", type=int, default=20, help="Dimension of model")
    parser.add_argument("--n_head", type=int, default=2, help="Number of heads")
    parser.add_argument("--dim_feedforward", type=int, default=20, help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--key_padding_mask", action='store_true', default=False, help="Enables key padding mask")
    parser.add_argument("--lr_init", type=float, default=0.001, help="Initial learning rate")
    
    parser.add_argument("--max_epochs", type=int, default=1, required=True, help="Maximum number of epochs")
    parser.add_argument("--max_time", default=None, help="Maximum time")
    parser.add_argument("--num_dataloader_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--protein_embeddings_path", default="data/embeddings/protein_embeddings.npy", help="Path to protein embeddings")
    parser.add_argument("--rna_embeddings_path", default="data/embeddings/rna_embeddings.npy", help="Path to RNA embeddings")
    parser.add_argument("--train_set_path", default='data/interactions/train_set.parquet', help="Path to the train set file")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--checkpoints_dir", default="checkpoints", help="Path to the checkpoints")
    parser.add_argument("--val_set_size", type=float, default=0.2, help="Size of the validation set")

    args = parser.parse_args()

    main(args)