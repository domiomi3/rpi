#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080  # partition (queue)
#SBATCH -t 0-06:00 # time (D-HH:MM)
#SBATCH -o /work/dlclarge1/gernel-RNAProteinModel/log/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/gernel-RNAProteinModel/log/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J RNAProteinInterAct  # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#SBATCH --chdir=/work/dlclarge1/gernel-RNAProteinModel/
#SBATCH --mem=180G
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


# Job to perform
/work/dlclarge1/gernel-RNAProteinModel/venv/bin/python main.py \
  --accelerator=gpu \
  --max-time="00:05:00:00" \
  --max-epochs=15 \
  --compiled \
  --wandb \
  --num-encoder-layers=1 \
  --key-padding-mask \
  --batch-size=8 \
  --d-model=4 \
  --n-head=2 \
  --lr-init=0.000001 \
  --dim-feedforward=8 \
  --protein-embeddings-path=/work/dlcsmall2/gernel-sample-ws/dataset_v2/protein_embeddings.npy \
  --rna-embeddings-path=/work/dlcsmall2/gernel-sample-ws/dataset_v2/rna_embeddings.npy \
  --db-file-valid=/work/dlcsmall2/gernel-sample-ws/dataset_v2/final_valid_set_reduced.parquet \
  --db-file-train=/work/dlcsmall2/gernel-sample-ws/dataset_v2/final_train_set_reduced.parquet \
  --dataloader-type=PandasInMemory \
  --num-dataloader-workers=8  # Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
