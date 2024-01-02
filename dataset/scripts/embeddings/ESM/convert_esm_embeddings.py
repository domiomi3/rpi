import numpy as np
import os
from time import time
from tqdm import tqdm
import psutil
import random

"""
Script helps to create single array for all protein embeddings. 
This array is required by src/data/dataloader class RNAInterActionsPandasInMemory
"""

def main():
    path = "/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/ESM/results"
    embeddings = []
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    paths = [(int(embed[:-4]), os.path.join(path, embed)) for embed in os.listdir(path) if embed.endswith('.npy')]
    paths = sorted(paths, key = lambda x: x[0])[:]
    for embed_id, embedding_path in tqdm(paths):
        arc = np.load(embedding_path)
        seq_2_embed = arc
        padded_seq_2_embed = np.zeros((1024, 640))
        padded_seq_2_embed[:seq_2_embed.shape[0], :] = seq_2_embed
        embeddings.append(padded_seq_2_embed)
    embeddings = np.stack(embeddings, axis=0)
    print(embeddings.shape)
    print("Done :)!")
    print("checking embeddings")
    check_array(embeddings, path, num_check=10)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print("Storing embeddings")
    np.save("/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/ESM/protein_embeddings.npy", embeddings)


def check_array(arr: np.array, path, num_check=100):
    num_embed = arr.shape[0]
    indexes = list(range(num_embed))
    check_indexes = random.choices(indexes, k=num_check)
    for idx in check_indexes:
        embed_path = os.path.join(path, f"{idx}.npy")
        seq_2_embed = np.load(embed_path)
        # breakpoint()
        padded_seq_2_embed = np.zeros((1024, 640))
        padded_seq_2_embed[:seq_2_embed.shape[0], :] = seq_2_embed
        assert (seq_2_embed[idx] == padded_seq_2_embed[idx]).all()


if __name__ == '__main__':
    main()