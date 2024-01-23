import sys
import argparse
import logging

import torch
import lightning.pytorch.loggers
import neps

from pathlib import Path

from lightning import Trainer

src_dir = Path.cwd().parent
sys.path.append(str(src_dir))
from model import RNAProteinInterAct, RNAProteinInterActSE, BaseCNN, ModelWrapper
from dataloader import get_dataloader


class Config:
    def __init__(self, **kwargs):
        self.d_model = 2048
        self.n_head = 2
        self.dim_feedforward = 20
        self.num_encoder_layers = 1
        self.key_padding_mask = True
        self.accelerator = 'cuda'
        self.devices = 1
        self.batch_size = 16
        self.one_hot_encoding = False
        self.baseline = False
        self.num_dataloader_workers = 8
        self.seed = 150
        self.loader_type = "RPIDataset"
        self.embedding_type = "new_esm_rnafm"
        self.protein_embeddings_path = "data/embeddings/protein_embeddings.npy"
        self.rna_embeddings_path = "data/embeddings/rna_embeddings.npy"
        self.train_set_path = "data/interactions/train_set.parquet"
        self.val_set_path = "data/interactions/validation_set.parquet"
        self.__dict__.update(kwargs)


def train_and_eval(pipeline_directory, previous_pipeline_directory, weight_decay, learning_rate, dropout, epochs):
    """
    Train and evaluate the RPI model based on provided hyperparameters and budget.
    For use with Hyperband strategy in neps.
    """

    # Load all used paramerters through Config class
    config = Config()

    # Set run name based on used config 
    config_parts = str(pipeline_directory).split("/")[-1].split("_")
    config_name = '_'.join(config_parts[:2])
    wandb_run_name = \
    f"{config.embedding_type} {config_name}: learning_rate={learning_rate:.5f}, \
    weight_decay={weight_decay:.5f}, dropout={dropout:.5f}"

    # Choose model based on embedding strategy
    if config.one_hot_encoding:
        model = RNAProteinInterActSE
    else:
        if config.baseline:
            rpi_model = BaseCNN(
                d_model=config.d_model,
                device=config.accelerator
            )
        else:
            model = RNAProteinInterAct

    if not config.baseline:
        # Initialize model
        rpi_model = model( 
            batch_first=True,
            embed_dim=640,
            d_model=config.d_model,
            num_encoder_layers=config.num_encoder_layers,
            nhead=config.n_head,
            dim_feedforward=config.dim_feedforward,
            key_padding_mask=config.key_padding_mask,
            norm_first=True,
            dropout=dropout
        )

    # Wrap the model in a Lightning module
    lightning_module = ModelWrapper(
        rpi_model,
        lr_init=learning_rate,
        weight_decay=weight_decay,
        seed=config.seed,
    )

    # Get optimizer
    optimizer = lightning_module.configure_optimizers()

    checkpoint_name = "checkpoint.pth"

    # Resume training if config has already been evaluated on lower budget
    if previous_pipeline_directory is not None:
        checkpoint = torch.load(previous_pipeline_directory / checkpoint_name)
        rpi_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epochs_previously_spent = checkpoint["epoch"]
        previous_wandb_run_id = checkpoint.get("wandb_run_id", None) 
    else:
        epochs_previously_spent = 0
        previous_wandb_run_id = None
    
    # Initialize logger
    logger = lightning.pytorch.loggers.WandbLogger(
        project="RNAProteinInterAct",
        name=wandb_run_name,
        id=previous_wandb_run_id,  # Resume the run if ID is provided
        resume="allow",  # Set to 'allow' to resume the run if ID is provided
    )

    # Set remaining number of epochs based on previous budget
    epochs_to_run = epochs - epochs_previously_spent  

    # Initialize trainer with min epochs based on previous budget
    trainer = Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        min_epochs=epochs_previously_spent,
        max_epochs=epochs,
        logger=logger,
        enable_checkpointing=False # checkpointing is handled by neps
    )

    # Get dataloaders
    train_dataloader = get_dataloader(  
        loader_type=config.loader_type,
        dataset_path=config.train_set_path,
        rna_embeddings_path=config.rna_embeddings_path,
        protein_embeddings_path=config.protein_embeddings_path,
        seed=config.seed,
        num_workers=config.num_dataloader_workers,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_dataloader = get_dataloader(  
        loader_type=config.loader_type,
        dataset_path=config.val_set_path,
        rna_embeddings_path=config.rna_embeddings_path,
        protein_embeddings_path=config.protein_embeddings_path,
        seed=config.seed,
        num_workers=config.num_dataloader_workers,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Train model
    trainer.fit(model=lightning_module, train_dataloaders=train_dataloader)

    # Save model
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": rpi_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "wandb_run_id": logger.experiment.id,
        },
        pipeline_directory / checkpoint_name,
    )

    # Evaluate model
    results = trainer.validate(lightning_module, val_dataloader)[0]
    loss = float(results['valid_loss_epoch'])

    logger.experiment.finish()

    return dict(loss=loss, cost=epochs_to_run)
    

def main(args):
    logging.basicConfig(level=logging.INFO)

    hp_space = dict(
        learning_rate=neps.FloatParameter(lower=1e-5, upper=1e-2, log=True),
        weight_decay=neps.FloatParameter(lower=1e-5, upper=1e-2, default=5e-4, log=True),
        dropout=neps.FloatParameter(lower=0, upper=1, default=0.2),
        epochs=neps.IntegerParameter(lower=3, upper=30, is_fidelity=True)
    )

    neps.run(
        run_pipeline=train_and_eval,
        pipeline_space=hp_space,
        root_directory=args.results_dir,
        searcher="hyperband",
        post_run_summary=True,
        max_cost_total=args.max_budget,
        overwrite_working_directory=True,
    )

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default="neps_results", help="Directory to store neps run results in")
    parser.add_argument("--max_budget", type=int, default=90, help="Maximum budget for neps run")
    args = parser.parse_args()

    main(args)