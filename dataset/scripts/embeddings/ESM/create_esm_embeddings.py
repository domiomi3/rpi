import torch
import esm
from time import time
from statistics import mean
import click
import numpy
import argparse
import pandas as pd
from more_itertools import divide

"""
This script helps to create ESM embeddings for proteins. 
It can be run with a array batch job SLURM cluster. 
e.g. pass SLURM_ARRAY_TASK_ID to --task-id
and SLURM_ARRAY_TASK_MAX to --max-task-id
This will distribute all protein sequences across different jobs.
Recommended to use with pre-trained model: esm2_t30_150M_UR50D ==> --repr-layer=30
(see https://github.com/facebookresearch/esm#available-models-and-datasets-) 
"""


# @click.command()
# @click.option('--enable-cuda', default=True)
# @click.option('--model-name', default="dd")
# @click.option('--protein-path', default="/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/results/unique_proteins.parquet")
# @click.option('--repr-layer', default=30)
# @click.option('--max-task-id', default=0)
# @click.option('--task-id', default=0)
# @click.option('--save-dir', default="results")
# @click.option('--log-freq', default=1000)

def divide_dataframe(df, max_task_id, task_id):
    """
    Divide a DataFrame into equal parts based on max_task_id and return the part specified by task_id.

    :param df: pandas DataFrame to be divided.
    :param max_task_id: The total number of parts to divide the DataFrame into.
    :param task_id: The index of the part to return (0-indexed).
    :return: The subset of the DataFrame corresponding to the task_id.
    """
    # Calculate the number of rows per task
    rows_per_task = len(df) // max_task_id

    # Calculate the start and end index for the slice
    start_idx = task_id * rows_per_task
    end_idx = start_idx + rows_per_task

    # Adjust the end index for the last task to include any remaining rows
    if task_id == max_task_id - 1:
        end_idx = len(df)

    # Return the subset of the DataFrame
    return df.iloc[start_idx:end_idx].to_dict('records')

#using GPU leads to OOM issues
def main(enable_cuda=False, model_name="dfs", protein_path="/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/results/unique_proteins.parquet", repr_layer=30, max_task_id=0, task_id=0, save_dir="/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/ESM/results", log_freq=1000):
    
    print(f"Running with task id {task_id} and max task id {max_task_id}")

    proteins_df = pd.read_parquet(protein_path, engine='pyarrow')
    # This line divides the dataframe into (max-task-id) parts and takes the (task-id)th element.
    data = divide_dataframe(proteins_df, max_task_id, task_id)
    # data = proteins_df.iloc[
    #     [list(c) for c in divide(max_task_id + 1, range(proteins_df.shape[0]))][task_id]].to_dict('records')
    print(f"Create embeddings for {len(data)} proteins out of {proteins_df.shape[0]} proteins from dataset {protein_path}")
    print(f"Model name: {model_name}")
    # check how much memory data occupies
    # print(f"Data size: {round(data.memory_usage().sum(), 2)}")

    # breakpoint()

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    # model, alphabet = esm.pretrained.load_model_and_alphabet_local(f"models/{model_name}.pt")
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    print("Model loaded.")
    print("Start to create embeddings.")
    timings = []

    # data = [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")]
    if enable_cuda:
        model.cuda()

    for idx, protein in enumerate(data):
        print(f"Processing protein {idx}/{len(data)}")
        if enable_cuda:
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mem = round(free_mem / 1024 / 1024 / 1024, 2)
            total_mem = round(total_mem / 1024 / 1024 / 1024, 2)
            used_mem = round(total_mem - free_mem, 2)
            print("GPU memory usage:")
            print(free_mem, total_mem, used_mem
            )
            # check deeper memory usage
            print(torch.cuda.memory_summary(device=None, abbreviated=True))
        start = time()
        protein['Sequence_2'] = protein['Sequence_2'].upper()
        protein_sequence = [(protein['Sequence_2_ID_Unique'], protein['Sequence_2'])]

        batch_labels, batch_strs, batch_tokens = batch_converter(protein_sequence)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        if enable_cuda:
            batch_tokens = batch_tokens.to(device='cuda')

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            #check size of batch_tokens
            print("Batch token: ", batch_tokens.size())
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=True)
        token_representations = results["representations"][repr_layer].cpu()
        # Generate per-sequence representations
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, tokens_len in enumerate(batch_lens):
            numpy.save(f"{save_dir}/{protein['Sequence_2_ID_Unique']}",
                    token_representations[i, 1: tokens_len - 1].float())

        del batch_tokens
        del token_representations
        del results
        del batch_labels
        del batch_strs
        del batch_lens
        del protein_sequence
        del protein

        if enable_cuda:
            torch.cuda.empty_cache()
            if idx % log_freq == 0:
                print(f"{idx}/{len(data)} embeddings done!")
                print(f"{used_mem}GB/{total_mem}GB used")
        timings.append(time() - start)

    print(f"Average time per protein embedding extraction: {round(mean(timings), 4)}")

# if __name__ == '__main__':
#     main()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--enable_cuda', type=bool, default=False, help='Enable or disable CUDA')
    parser.add_argument('--model_name', type=str, default='dd', help='Name of the model')
    parser.add_argument('--protein_path', type=str, default='/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/results/unique_proteins.parquet', help='Path to the protein data')
    parser.add_argument('--repr_layer', type=int, default=30, help='Representation layer')
    parser.add_argument('--max_task_id', type=int, default=20, help='Maximum task ID')
    parser.add_argument('--task_id', type=int, default=1, help='Task ID')
    parser.add_argument('--save_dir', type=str, default='/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/ESM/results', help='Directory to save the results')
    parser.add_argument('--log_freq', type=int, default=1000, help='Log frequency')

    args = parser.parse_args()

    main(args.enable_cuda, args.model_name, args.protein_path, args.repr_layer, args.max_task_id, args.task_id, args.save_dir, args.log_freq)
