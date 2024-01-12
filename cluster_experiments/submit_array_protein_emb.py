import logging

import submitit
 
logger = logging.getLogger("domi.omi")
 
 
def submit() -> None:
    executor = submitit.AutoExecutor(folder="/work/dlclarge1/matusd-rpi/RPI/cluster_logs")
    executor.update_parameters(
        partition="mlhiwidlc_gpu-rtx2080",
        slurm_time="1:00:00",
        slurm_mem="100000mb",
        slurm_cpus_per_task="3",
        slurm_job_name="protein_embeddings",
    )
 
    def job_function(args):
        import sys
        sys.path.append('/work/dlclarge1/matusd-rpi/RPI/dataset/embeddings')
        import esm_rna_fm 
        return esm_rna_fm.create_embeddings(**args)
 
    logger.info("Submitting jobs...")

    emb_dir = "/work/dlclarge1/matusd-rpi/RPI/data/embeddings/"
    protein_path = "/work/dlclarge1/matusd-rpi/RPI/data/embeddings/unique_proteins.parquet"
    model_type = "esm2"
    enable_cuda = False
    repr_layer = 30
    max_task_id = 20
    task_ids = list(range(0, 1))
    jobs_to_submit = []
 
    for task_id in task_ids:
        submit_args_exec = {
            "emb_dir": emb_dir,
            "protein_path": protein_path,
            "model_type": model_type,
            "enable_cuda": enable_cuda,
            "repr_layer": repr_layer,
            "task_id": task_id,
            "max_task_id": max_task_id,
        }
        jobs_to_submit.append(submit_args_exec)
 
    if jobs_to_submit:
        with executor.batch():
            for submit_args_exec in jobs_to_submit:
                _ = executor.submit(job_function, submit_args_exec)
                logger.info(f"Prepared job with args: {submit_args_exec}")
 
    logger.info("Submitted all jobs as an array job")
 
 
if __name__ == "__main__":
    submit()  # pylint: disable=no-value-for-parameter