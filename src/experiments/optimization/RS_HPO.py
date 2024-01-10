from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer
import traceback
import subprocess

"""
This script explores the search space via Random Search. 
Useful for initial exploration.
"""

# define search space
lr_init = Float("lr_init", (1e-10, 1e-1), default=1e-3, log=True)
d_model = Integer("d_model", (20, 200), default=100, q=20)
num_encoder_layers = Integer("num_encoder_layers", (1, 2), default=1)
feedforward_mul = Integer("feedforward_mul", (1, 4), default=2)
weight_decay = Float("weight_decay", (1e-8, 1e+0), default=1e-1, log=True)
dropout = Float("dropout", (0.01, 0.5), default=0.1)
cs = ConfigurationSpace()
cs.add_hyperparameters([lr_init, d_model, num_encoder_layers, feedforward_mul, weight_decay, dropout])
dataloader_type = "PandasInMemory"
db_file_train = "/work/dlcsmall2/gernel-sample-ws/dataset_v2/final_train_set_reduced.parquet"
db_file_valid = "/work/dlcsmall2/gernel-sample-ws/dataset_v2/final_valid_set_reduced.parquet"
rna_embeddings_path = "/work/dlcsmall2/gernel-sample-ws/dataset_v2/rna_embeddings.npy"
protein_embeddings_path = "/work/dlcsmall2/gernel-sample-ws/dataset_v2/protein_embeddings.npy"
num_dataloader_workers = 8
batch_size = 16


for idx in range(9999):
    config = dict(cs.sample_configuration())
    print(idx)
    print(config)
    try:
        args = [
            "/work/dlclarge1/gernel-RNAProteinModel/venv/bin/python",
            "train_rna-fm.py",
            "--accelerator", "gpu",
            "--devices", "1",
            "--compiled",
            "--wandb",
            "--num-encoder-layers", f"{config['num_encoder_layers']}",
            "--max-epochs", "10",
            "--num-dataloader-workers", "8",
            "--batch-size", "8",
            "--d-model", f"{config['d_model']}",
            "--n-head", f"2",
            "--dim-feedforward", f"{config['d_model'] * config['feedforward_mul']}",
            "--dropout", f"{config['dropout']}",
            "--weight-decay", f"{config['weight_decay']}",
            "--key-padding-mask",
            "--lr-init", f"{config['lr_init']}",
            "--dataloader-type", f"{dataloader_type}",
            "--protein-embeddings-path", f"{protein_embeddings_path}",
            "--rna-embeddings-path", f"{rna_embeddings_path}",
            "--db-file-train", f"{db_file_train}",
            "--db-file-valid", f"{db_file_valid}"
        ]
        subprocess.call(args)
    except Exception:
        print(traceback.format_exc())
        print("Something occured for config")
        print(config)

