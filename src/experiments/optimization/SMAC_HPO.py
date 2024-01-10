from ConfigSpace import (ConfigurationSpace,
                         Configuration,
                         Float,
                         Constant)

import numpy as np

from src.model import RNAProteinInterAct, ModelWrapper
from src.dataloader import RNAInterActionsPandasInMemory
from lightning.pytorch.callbacks import LearningRateMonitor
import torch

import lightning.pytorch.loggers
from lightning import Trainer
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import click
from statistics import mean
import wandb

"""
Runs HPO with SMAC (see https://automl.github.io/SMAC3/main/3_getting_started.html)
"""

class RPI:
    @property
    def get_configspace(self) -> ConfigurationSpace:

        # define search space
        lr_init = Float("lr_init", (1e-5, 1e-1), default=1e-3, log=True)
        weight_decay = Float("weight_decay", (1e-4, 1e-1), default=1e-1, log=True)
        dropout = Float("dropout", (0.01, 0.5), default=0.1)
        feedforward_mul = Constant("feedforward_mul", 1)
        d_model = Constant("d_model", 20)
        num_encoder_layers = Constant("num_encoder_layers", 1)
        # feedforward_mul = Integer("feedforward_mul", (1, 3), default=2)
        # d_model = Integer("d_model", (20, 60), default=20, q=10)
        # num_encoder_layers = Integer("num_encoder_layers", (1, 2), default=1)
        n_head = Constant("n_head", 2)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([lr_init, d_model, num_encoder_layers, feedforward_mul, weight_decay, dropout, n_head])
        return cs

    def train(self, config: Configuration,
              budget: int,
              accelerator: str,
              devices: str,
              compiled,
              wandb_log,
              dataset,
              num_dataloader_worker,
              batch_size,
              n_splits: int,
              seed: int = 0) -> float:

        rna_protein_model = RNAProteinInterAct(batch_first=True,
                                               embed_dim=640,
                                               d_model=config['d_model'],
                                               num_encoder_layers=config['num_encoder_layers'],
                                               nhead=config['n_head'],
                                               dim_feedforward=config['d_model'] * config['feedforward_mul'],
                                               key_padding_mask=True,
                                               norm_first=True,
                                               dropout=config['dropout']
                                               )
        lightning_module = ModelWrapper(rna_protein_model,
                                        lr_init=config['lr_init'],
                                        weight_decay=config['weight_decay'],
                                        seed=seed
                                        )
        if compiled:
            lightning_module = torch.compile(lightning_module)
        # True is the default value of the logger.
        logger = True
        if wandb_log:
            logger = lightning.pytorch.loggers.WandbLogger(project="RNAProteinInterAct", tags=['SMAC-v2'])
            logger.log_hyperparams(dict(config))
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Returns the 3-fold cross validation accuracy
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)  # to make CV splits consistent
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
            trainer = Trainer(accelerator=accelerator,
                              devices=devices,
                              max_epochs=int(np.ceil(budget)),
                              logger=logger,
                              enable_progress_bar=False,
                              callbacks=[lr_monitor])
            print(f"Fold {fold + 1}")
            print("-------")

            # Define the data loaders for the current fold
            train_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(train_idx),
                num_workers=num_dataloader_worker,
            )
            test_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(test_idx),
                num_workers=num_dataloader_worker
            )
            trainer.fit(lightning_module, train_loader)
            results = trainer.validate(lightning_module, test_loader)[0]
            fold_results.append(results['val_BinaryF1Score'])
            if wandb_log:
                logger.experiment.finish()
                wandb.finish()
                logger = lightning.pytorch.loggers.WandbLogger(project="RNAProteinInterAct", tags=['SMAC'])

        return 1 - mean(fold_results)


@click.command()
@click.option("--accelerator", default='cpu')
@click.option("--devices", default=1)
@click.option("--compiled", is_flag=True, show_default=True, default=False, help="Enables Torch compiled Module.")
@click.option("--wandb-log", is_flag=True, show_default=True, default=False, help="Enables logging via wandb.")
@click.option("--num-dataloader-workers", default=2)
@click.option("--batch-size", default=2)
@click.option("--n-splits", default=2)
@click.option("--min-budget", default=3)
@click.option("--max-budget", default=20)
@click.option("--n-configs", default=2)
@click.option("--protein-embeddings-path", default="dataset/protein_embeddings")
@click.option("--rna-embeddings-path", default="dataset/rna_embeddings")
@click.option("--db-file-train", default='dataset/final_train_set.parquet')
@click.option("--walltime-limit", default=43200)
def main(accelerator,
         devices,
         compiled,
         wandb_log,
         db_file_train,
         rna_embeddings_path,
         protein_embeddings_path,
         num_dataloader_workers,
         n_splits,
         n_configs,
         min_budget,
         max_budget,
         batch_size,
         walltime_limit):
    # get dataloader

    rna_embeddings = RNAInterActionsPandasInMemory.pre_load_rna_embeddings(
        rna_embeddings_path
    )
    protein_embeddings = RNAInterActionsPandasInMemory.pre_load_protein_embeddings(
        protein_embeddings_path
    )
    dataset = RNAInterActionsPandasInMemory(
        rna_embeddings,
        protein_embeddings,
        db_file=db_file_train
    )

    rpi = RPI()
    intensifier_object = Hyperband

    # define scenario
    scenario = Scenario(
        rpi.get_configspace,
        walltime_limit=walltime_limit,  # Stop HPO after 12 hours
        n_trials=500,  # evaluate max 500 trials
        min_budget=min_budget,  # at least 5 epochs
        max_budget=max_budget,  # max 20 epochs,
        n_workers=1
    )

    f_run_cfg = lambda c, seed, budget: rpi.train(c, budget, accelerator, devices, compiled, wandb_log,
                                                  dataset,
                                                  num_dataloader_workers,
                                                  batch_size, n_splits,
                                                  seed)

    # run random configs at beginning
    initial_design = MFFacade.get_initial_design(scenario,
                                                 n_configs=n_configs,
                                                 )

    # create intensifier
    intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")

    # create SMAC object
    smac = MFFacade(
        scenario,
        f_run_cfg,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
    )

    smac.optimize()
    # incumbent (=best performing hyperparameters) are stored to log files


if __name__ == '__main__':
    main()
