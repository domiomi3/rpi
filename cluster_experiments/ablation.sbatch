#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080  # partition (queue)
#SBATCH -t 0-03:30 # time (D-HH:MM)
#SBATCH -o /work/dlclarge1/gernel-RNAProteinModel/log/%A.%a.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/gernel-RNAProteinModel/log/%A.%a.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J RNAProteinInterAct  # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#SBATCH --chdir=/work/dlclarge1/gernel-RNAProteinModel/
#SBATCH --mem=180G
#SBATCH --array=7%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

random_loaders=(
        "PIMAllRandom"
        "PIMRNARandom"
        "PIMProteinRandom"
        "PIMAllZero"
        "PIMRNAZero"
        "PIMProteinZero"
        "PIMAllRNA"
        "PIMAllProtein"
)
random_loader=${random_loaders[$SLURM_ARRAY_TASK_ID]}
# Job to perform
/work/dlclarge1/gernel-RNAProteinModel/venv/bin/python main.py \
  --accelerator=gpu \
  --max-epochs=10 \
  --compiled \
  --wandb \
  --num-encoder-layers=1 \
  --batch-size=128 \
  --protein-embeddings-path=/work/dlcsmall2/gernel-sample-ws/protein_embeddings_float32.npy \
  --rna-embeddings-path=/work/dlcsmall2/gernel-sample-ws/rna_embeddings_float32.npy \
  --dataloader-type="$random_loader" \
  --num-dataloader-workers=8  # Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";