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
        sys.path.append('/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/ESM')
        import create_esm_embeddings 
        return create_esm_embeddings.main(task_id=args['task_id'], max_task_id=args['max_task_id'])
 
    logger.info("Submitting jobs...")
    # breakpoint()

    max_task_id = 20
    task_ids = list(range(0, max_task_id))
    jobs_to_submit = []
 
    for task_id in task_ids:
        submit_args_exec = {
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