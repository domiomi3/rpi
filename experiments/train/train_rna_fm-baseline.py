import lightning.pytorch.loggers
from lightning import Trainer
import torch
from src.models.model import BaseLinear, ModelWrapper, BaseCNN
import click
from src.data.dataloader import get_dataloader
from lightning.pytorch.callbacks import LearningRateMonitor

"""
This script trains baseline models using RNA-FM & ESM embeddings.
"""

@click.command()
@click.option("--accelerator", default='cpu')
@click.option("--devices", default=1)
@click.option("--compiled", is_flag=True, show_default=True, default=False, help="Enables Torch compiled Module.")
@click.option("--wandb", is_flag=True, show_default=True, default=False, help="Enables logging via wandb.")
@click.option("--max-epochs", default=-1)
@click.option("--max-time", default=None, type=str)
@click.option("--num-dataloader-workers", default=2)
@click.option("--batch-size", default=16)
@click.option("--weight-decay", default=0.1)
@click.option("--lr-init", default=0.001)
@click.option("--dataloader-type", default="Pandas")
@click.option("--model", default="Linear")
@click.option("--protein-embeddings-path", default="dataset/protein_embeddings")
@click.option("--rna-embeddings-path", default="dataset/rna_embeddings")
@click.option("--db-file-train", default='dataset/final_train_set.parquet')
@click.option("--db-file-valid", default='dataset/final_valid_set.parquet')
def main(accelerator: str, devices: int,
         compiled: bool, max_epochs: int, max_time: str,
         wandb: bool,
         num_dataloader_workers: int, batch_size: int,
         lr_init: float,
         dataloader_type: str,
         model: str,
         protein_embeddings_path: str, rna_embeddings_path: str,
         weight_decay: float,
         db_file_train: str, db_file_valid: str):
    rna_protein_model = None
    if model == 'CNN':
        rna_protein_model = BaseCNN(
            d_model=1174
        )
    if model == 'Linear':
        rna_protein_model = BaseLinear(
            d_model=640
        )
    assert rna_protein_model is not None
    lightning_module = ModelWrapper(rna_protein_model,
                                    lr_init=lr_init,
                                    weight_decay=weight_decay
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
    train_dataloader, valid_dataloader = get_dataloader(dataloader_type,
                                                        db_file_train,
                                                        db_file_valid,
                                                        rna_embeddings_path,
                                                        protein_embeddings_path,
                                                        num_workers=num_dataloader_workers,
                                                        batch_size=batch_size
                                                        )

    trainer.fit(lightning_module, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    main()