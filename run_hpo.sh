#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=100000mb
#SBATCH --job-name=hpo_new_esm_rnafm
#SBATCH --chdir=/work/dlclarge1/matusd-rpi/RPI
#SBATCH --output="LOGS//%x.%N.%A.%a.out"
#SBATCH --error="LOGS//%x.%N.%A.%a.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matus.dominika@gmail.com

echo "Working dir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /home/matusd/.conda/bin/activate rpi

workdir=/work/dlclarge1/matusd-rpi/RPI

cd "$workdir"

python src/hpo.py --results_dir "neps_results/neps_results/esm_rnafm_new" --max_budget 90