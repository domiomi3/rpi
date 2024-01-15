#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00
#SBATCH --mem=100000mb
#SBATCH --job-name=rpi_testing
#SBATCH --output="LOGS//%x.%N.%A.%a.out"
#SBATCH --error="LOGS//%x.%N.%A.%a.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matus.dominika@gmail.com

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /home/matusd/.conda/bin/conda/activate rpi

workdir=/work/dlclarge1/matusd-rpi/RPI

cd "$workdir"

python experiments/train/train_rna-fm_random_split.py \