#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080  # partition (queue)
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -o /work/dlclarge1/gernel-RNAProteinModel/log/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/gernel-RNAProteinModel/log/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J Final-RNA-FM0  # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#SBATCH --chdir=/work/dlclarge1/gernel-RNAProteinModel/
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


# Job to perform
/work/dlclarge1/gernel-RNAProteinModel/venv/bin/python /work/dlclarge1/gernel-RNAProteinModel/final-rna-fm.py --accelerator gpu --devices 1 --wandb --num-encoder-layers 1 --max-epochs 150 --num-dataloader-workers 8 --batch-size 8 --d-model 20 --n-head 2 --dim-feedforward 20 --dropout 0.21759085167606332 --weight-decay 0.00022637229697395497 --key-padding-mask --lr-init 0.00001923730509654649 --dataloader-type PandasInMemory --protein-embeddings-path /work/dlcsmall2/gernel-sample-ws/dataset_v4/protein_embeddings.npy --rna-embeddings-path /work/dlcsmall2/gernel-sample-ws/dataset_v4/rna_embeddings.npy --db-file-train /work/dlcsmall2/gernel-sample-ws/dataset_v4/final/train_set.parquet --db-file-valid /work/dlcsmall2/gernel-sample-ws/dataset_v4/final/test_set_random.parquet --seed=0
echo "DONE";
echo "Finished at $(date)";
