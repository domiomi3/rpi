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
    print("Done")
    # TODO check embedding sizes
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print("Storing embeddings.")
    np.save("/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/ESM/protein_embeddings.npy", embeddings)




if __name__ == '__main__':
    main()