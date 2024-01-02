import lightning.pytorch.loggers
from lightning import Trainer
import torch
import sys
sys.path.append('/work/dlclarge1/matusd-rpi/RPI/src')
sys.path.append('/work/dlclarge1/matusd-rpi/RPI/src/models')
sys.path.append('/work/dlclarge1/matusd-rpi/RPI/src/data')
from models.model import RNAProteinInterAct, ModelWrapper
import click
from data.dataloader import get_random_dataloader
from lightning.pytorch.callbacks import LearningRateMonitor

import argparse
"""
This script trains the model using RNA-FM & ESM embeddings. 
Train and Validation are based on a random split!
"""

# @click.command()
# @click.option("--accelerator", default='cpu')
# @click.option("--devices", default=1)
# @click.option("--compiled", is_flag=True, show_default=True, default=False, help="Enables Torch compiled Module.")
# @click.option("--wandb", is_flag=True, show_default=True, default=False, help="Enables logging via wandb.")
# @click.option("--num-encoder-layers", default=6)
# @click.option("--max-epochs", default=-1)
# @click.option("--max-time", default=None, type=str)
# @click.option("--num-dataloader-workers", default=2)
# @click.option("--batch-size", default=16)
# @click.option("--d-model", default=320)
# @click.option("--n-head", default=2)
# @click.option("--dropout", default=0.1)
# @click.option("--weight-decay", default=0.1)
# @click.option("--dim-feedforward", default=640)
# @click.option("--key-padding-mask", is_flag=True, show_default=True, default=False, help="Enables key padding mask")
# @click.option("--lr-init", default=0.001)
# @click.option("--dataloader-type", default="Pandas")
# @click.option("--protein-embeddings-path", default="dataset/protein_embeddings")
# @click.option("--rna-embeddings-path", default="dataset/rna_embeddings")
# @click.option("--db-file-train", default='dataset/final_train_set.parquet')
# @click.option("--db-file-valid", default='dataset/final_valid_set.parquet')
def main(accelerator: str, devices: int,
         compiled: bool, max_epochs: int, max_time: str,
         wandb: bool, num_encoder_layers: int,
         num_dataloader_workers: int, batch_size: int,
         d_model: int, n_head: int,
         dim_feedforward: int, lr_init: float,
         key_padding_mask: bool, dataloader_type: str,
         protein_embeddings_path: str, rna_embeddings_path: str,
         dropout: float, weight_decay: float,
         db_file_train: str, db_file_valid: str,
         seed: int):
    rna_protein_model = RNAProteinInterAct(batch_first=True,
                                           embed_dim=640,
                                           d_model=d_model,
                                           num_encoder_layers=num_encoder_layers,
                                           nhead=n_head,
                                           dim_feedforward=dim_feedforward,
                                           key_padding_mask=key_padding_mask,
                                           norm_first=True,
                                           dropout=dropout
                                           )
    lightning_module = ModelWrapper(rna_protein_model,
                                    lr_init=lr_init,
                                    weight_decay=weight_decay,
                                    seed=seed
                                    )
    if compiled:
        lightning_module = torch.compile(lightning_module)
    # True is the default value of the logger.
    logger = True
    if wandb:
        logger = lightning.pytorch.loggers.WandbLogger(project="RNAProteinInterAct")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(accelerator=accelerator,
                      devices=devices,
                      max_epochs=max_epochs,
                      max_time=max_time,
                      logger=logger,
                      callbacks=[lr_monitor])
    train_dataloader, valid_dataloader = get_random_dataloader(
                                        dataloader_type,
                                        db_file_train,
                                        db_file_valid,
                                        rna_embeddings_path,
                                        protein_embeddings_path,
                                        num_workers=num_dataloader_workers,
                                        batch_size=batch_size
    )
    trainer.fit(lightning_module, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script description here')

    parser.add_argument("--accelerator", default='cpu', type=str, help="Specify the accelerator.")
    parser.add_argument("--devices", default=1, type=int, help="Number of devices to use.")
    parser.add_argument("--compiled", action='store_true', help="Enables Torch compiled Module.")
    parser.add_argument("--wandb", action='store_true', help="Enables logging via wandb.")
    parser.add_argument("--num-encoder-layers", default=6, type=int, help="Number of encoder layers.")
    parser.add_argument("--max-epochs", default=-1, type=int, help="Maximum number of epochs.")
    parser.add_argument("--max-time", default=None, type=str, help="Maximum time for training.")
    parser.add_argument("--num-dataloader-workers", default=2, type=int, help="Number of dataloader workers.")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size.")
    parser.add_argument("--d-model", default=320, type=int, help="Dimension of the model.")
    parser.add_argument("--n-head", default=2, type=int, help="Number of heads.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
    parser.add_argument("--weight-decay", default=0.1, type=float, help="Weight decay rate.")
    parser.add_argument("--dim-feedforward", default=640, type=int, help="Dimension of feedforward network.")
    parser.add_argument("--key-padding-mask", action='store_true', help="Enables key padding mask.")
    parser.add_argument("--lr-init", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--dataloader-type", default="Pandas", type=str, help="Type of dataloader.")
    parser.add_argument("--protein-embeddings-path", default="/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/ESM/results/protein_embeddings.npy", type=str, help="Path to protein embeddings.")
    parser.add_argument("--rna-embeddings-path", default="/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/RNA-FM/results/rna_embeddings.npy", type=str, help="Path to RNA embeddings.")
    parser.add_argument("--db-file-train", default='/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/results/final_train_set.parquet', type=str, help="Path to training database file.")
    parser.add_argument("--db-file-valid", default='/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/results/final_test_set_random.parquet', type=str, help="Path to validation database file.")
    parser.add_argument("--seed", default=0, type=int, help="Seed for reproducibility.")
    args = parser.parse_args()
    main(args.accelerator, args.devices, args.compiled, args.max_epochs, args.max_time, args.wandb, args.num_encoder_layers, args.num_dataloader_workers, args.batch_size, args.d_model, args.n_head, args.dim_feedforward, args.lr_init, args.key_padding_mask, args.dataloader_type, args.protein_embeddings_path, args.rna_embeddings_path, args.dropout, args.weight_decay, args.db_file_train, args.db_file_valid, args.seed)