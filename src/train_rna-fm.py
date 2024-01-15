import argparse
import sys

import torch
import lightning.pytorch.loggers

from pathlib import Path
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor

src_dir = Path.cwd().parent
sys.path.append(str(src_dir))
from model import RNAProteinInterAct, ModelWrapper
from dataloader import get_dataloader

def main(args):
    torch.manual_seed(args.seed)

    rna_protein_model = RNAProteinInterAct(batch_first=True,
                                           embed_dim=640,
                                           d_model=args.d_model,
                                           num_encoder_layers=args.num_encoder_layers,
                                           nhead=args.n_head,
                                           dim_feedforward=args.dim_feedforward,
                                           key_padding_mask=args.key_padding_mask,
                                           norm_first=True,
                                           dropout=args.dropout
                                           )
    
    lightning_module = ModelWrapper(rna_protein_model,
                                    lr_init=args.lr_init,
                                    weight_decay=args.weight_decay,
                                    seed=args.seed
                                    )
    if args.compiled:
        lightning_module = torch.compile(lightning_module)
    
    logger = True
    if args.wandb:
        logger = lightning.pytorch.loggers.WandbLogger(project="RNAProteinInterAct")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(accelerator=args.accelerator,
                      devices=args.devices,
                      max_epochs=args.max_epochs,
                      max_time=args.max_time,
                      logger=logger,
                      callbacks=[lr_monitor])

    train_dataloader, valid_dataloader = get_dataloader(args.dataloader_type,
                                                        args.db_file_train,
                                                        args.rna_embeddings_path,
                                                        args.protein_embeddings_path,
                                                        num_workers=args.num_dataloader_workers,
                                                        batch_size=args.batch_size,
                                                        )

    # train_dataloader.__getitem__(0)

    trainer.fit(lightning_module, train_dataloader, valid_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RNA-Protein Interaction Model')

    parser.add_argument("--accelerator", default='gpu', type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--compiled", action='store_true', help="Enables Torch compiled Module.")
    parser.add_argument("--wandb", action='store_true', help="Enables logging via wandb.")
    parser.add_argument("--num-encoder-layers", default=1, type=int)
    parser.add_argument("--max-epochs", default=1, type=int)
    parser.add_argument("--max-time", default=None, type=str)
    parser.add_argument("--num-dataloader-workers", default=8, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--d-model", default=20, type=int)
    parser.add_argument("--n-head", default=2, type=int)
    parser.add_argument("--dropout", default=0.21759085167606332, type=float)
    parser.add_argument("--weight-decay", default=0.00022637229697395497, type=float)
    parser.add_argument("--dim-feedforward", default=20, type=int)
    parser.add_argument("--key-padding-mask", action='store_true', help="Enables key padding mask")
    parser.add_argument("--lr-init", default=0.00001923730509654649, type=float)
    parser.add_argument("--dataloader-type", default="PandasInMemory", type=str)
    parser.add_argument("--protein-embeddings-path", default="data/embeddings/protein_embeddings.npy", type=str)
    parser.add_argument("--rna-embeddings-path", default="data/embeddings/rna_embeddings.npy", type=str)
    parser.add_argument("--db-file-train", default='data/interactions/train_set.parquet', type=str)
    parser.add_argument("--seed", default=55, type=int)
    
    args = parser.parse_args()

    main(args)
