#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00
#SBATCH --mem=100000mb
#SBATCH --job-name=rpi_random_split
#SBATCH --output="experiment_logs//%x.%N.%A.%a.out"
#SBATCH --error="experiment_logs//%x.%N.%A.%a.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matus.dominika@gmail.com

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /home/matusd/.conda/bin/conda/activate rpi

workdir=/work/dlclarge1/matusd-rpi/RPI

cd "$workdir"

python experiments/train/train_rna-fm_random_split.py \
--accelerator gpu \
--devices 1 \
--wandb \
--num-encoder-layers 1 \
--max-epochs 150 \
--num-dataloader-workers 8 \
--batch-size 8 \
--d-model 20 \
--n-head 2 \
--dim-feedforward 20 \
--dropout 0.21759085167606332 \
--weight-decay 0.00022637229697395497 \
--key-padding-mask \
--lr-init 0.00001923730509654649 \
--dataloader-type PandasInMemory \
--protein-embeddings-path dataset/scripts/embeddings/ESM/protein_embeddings.npy \
--rna-embeddings-path dataset/scripts/embeddings/RNA-FM/rna_embeddings.npy \
--db-file-train dataset/scripts/annotate/dataset/results/final_train_set.parquet \
--db-file-valid dataset/scripts/annotate/dataset/results/final_test_set_random.parquet \
--seed=0