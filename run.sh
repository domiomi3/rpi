#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=100000mb
#SBATCH --job-name=rpi_testing_2
#SBATCH --chdir=/work/dlclarge1/matusd-rpi/RPI
#SBATCH --output="LOGS//%x.%N.%A.%a.out"
#SBATCH --error="LOGS//%x.%N.%A.%a.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matus.dominika@gmail.com

echo "Working dir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /home/matusd/.conda/bin/conda/activate rpi

workdir=/work/dlclarge1/matusd-rpi/RPI

cd "$workdir"

python src/train.py \
--accelerator gpu \
--devices 1 \
--wandb \
--num_encoder_layers 1 \
--max_epochs 2 \
--num_dataloader_workers 8 \
--batch_size 8 \
--d_model 20 \
--n_head 2 \
--dim_feedforward 20 \
--dropout 0.21759085167606332 \
--weight_decay 0.00022637229697395497 \
--key_padding_mask \
--lr_init 0.00001923730509654649 \
--dataloader_type PandasInMemory \
--protein_embeddings_path data/embeddings/protein_embeddings.npy \
--rna_embeddings_path data/embeddings/rna_embeddings.npy \
--train_set_path data/interactions/train_set.parquet \
--val_set_size 0.15 \
--seed 0 \
