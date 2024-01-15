import lightning.pytorch.loggers
from lightning import Trainer
import torch
from torch.utils.data import DataLoader

from src.model import RNAProteinInterAct, ModelWrapper, RNAProteinInterActRNAFormer
import click
from src.dataloader import RNAInterActionsPandasInMemory
from lightning.pytorch.callbacks import LearningRateMonitor
import json
import pathlib

"""
This script trains our model with predefined hyperparameters and evaluates it on different datasets.
Possible to run with RNA-FM or RNAFormer embeddings
"""


@click.command()
@click.option("--accelerator", default='cpu')
@click.option("--devices", default=1)
@click.option("--compiled", is_flag=True, show_default=True, default=False, help="Enables Torch compiled Module.")
@click.option("--wandb", is_flag=True, show_default=True, default=False, help="Enables logging via wandb.")
@click.option("--num-encoder-layers", default=6)
@click.option("--max-epochs", default=-1)
@click.option("--max-time", default=None, type=str)
@click.option("--num-dataloader-workers", default=2)
@click.option("--batch-size", default=16)
@click.option("--d-model", default=320)
@click.option("--n-head", default=2)
@click.option("--dropout", default=0.1)
@click.option("--weight-decay", default=0.1)
@click.option("--dim-feedforward", default=640)
@click.option("--key-padding-mask", is_flag=True, show_default=True, default=False, help="Enables key padding mask")
@click.option("--lr-init", default=0.001)
@click.option("--dataloader-type", default="Pandas")
@click.option("--protein-embeddings-path", default="dataset/protein_embeddings")
@click.option("--rna-embeddings-path", default="dataset/rna_embeddings")
@click.option("--db-file-train", default='dataset/final_train_set.parquet')
@click.option("--db-file-valid", multiple=True, default=['dataset/final_valid_set.parquet'])
@click.option("--rpi-protein-embeddings-path", default="dataset/rpi_protein_embeddings")
@click.option("--rpi-rna-embeddings-path", default="dataset/rpi_rna_embeddings")
@click.option("--db-file-rpi", default="dataset/rpi2825_extended.parquet")
@click.option("--results-dir", default="final_results")
@click.option("--rna-type", default="RNA-FM")
@click.option("--seed", default=0)
def main(accelerator: str, devices: int,
         compiled: bool, max_epochs: int, max_time: str,
         wandb: bool, num_encoder_layers: int,
         num_dataloader_workers: int, batch_size: int,
         d_model: int, n_head: int,
         dim_feedforward: int, lr_init: float,
         key_padding_mask: bool, dataloader_type: str,
         protein_embeddings_path: str, rna_embeddings_path: str,
         dropout: float, weight_decay: float,
         db_file_train: str, db_file_valid: list,
         rpi_protein_embeddings_path: str, rpi_rna_embeddings_path: str,
         db_file_rpi: str, results_dir,
         rna_type: str,
         seed: int):
    rna_protein_model = None
    if rna_type == 'rna-fm':
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
    elif rna_type == 'RNAFormer':
        rna_protein_model = RNAProteinInterActRNAFormer(batch_first=True,
                                               embed_dim=640,
                                               d_model=d_model,
                                               num_encoder_layers=num_encoder_layers,
                                               nhead=n_head,
                                               dim_feedforward=dim_feedforward,
                                               key_padding_mask=key_padding_mask,
                                               norm_first=True,
                                               dropout=dropout
                                               )
    assert rna_protein_model is not None, f"{rna_type} is invalid"
    lightning_module = ModelWrapper(model=rna_protein_model,
                                    lr_init=lr_init,
                                    weight_decay=weight_decay, seed=seed
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
                      enable_progress_bar=False,
                      callbacks=[lr_monitor])
    rna_embeddings = RNAInterActionsPandasInMemory.pre_load_rna_embeddings(
        rna_embeddings_path
    )
    protein_embeddings = RNAInterActionsPandasInMemory.pre_load_protein_embeddings(
        protein_embeddings_path
    )
    train_dataset = RNAInterActionsPandasInMemory(
        rna_embeddings,
        protein_embeddings,
        db_file=db_file_train
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=num_dataloader_workers,
                                  batch_size=batch_size)
    trainer.fit(lightning_module, train_dataloader)

    train_dataloader = None
    del train_dataloader

    for db_file in db_file_valid:
        print("Loading", db_file)
        dataset = RNAInterActionsPandasInMemory(
            rna_embeddings,
            protein_embeddings,
            db_file=db_file
        )
        dataloader = DataLoader(dataset, shuffle=False, num_workers=num_dataloader_workers,
                                batch_size=batch_size)

        results = trainer.validate(lightning_module, dataloader)
        pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
        save_path = pathlib.Path(results_dir).joinpath(f"{rna_type}-{pathlib.Path(db_file).stem}-{seed}.json")
        with open(save_path, "w") as write_file:
            json.dump(results, write_file, indent=4)

    rna_embeddings_rpi = RNAInterActionsPandasInMemory.pre_load_rna_embeddings(
        rpi_rna_embeddings_path
    )
    protein_embeddings_rpi = RNAInterActionsPandasInMemory.pre_load_protein_embeddings(
        rpi_protein_embeddings_path
    )
    print("Loading ", db_file_rpi)
    rpi2825_dataset = RNAInterActionsPandasInMemory(
        rna_embeddings_rpi,
        protein_embeddings_rpi,
        db_file=db_file_rpi
    )
    rpi2825_dataloader = DataLoader(rpi2825_dataset, shuffle=False, num_workers=num_dataloader_workers,
                                    batch_size=batch_size)
    results = trainer.validate(lightning_module, rpi2825_dataloader)
    save_path = pathlib.Path(results_dir).joinpath(f"{rna_type}-{pathlib.Path(db_file_rpi).stem}-{seed}.json")
    with open(save_path, "w") as write_file:
        json.dump(results, write_file, indent=4)


if __name__ == '__main__':
    main()
