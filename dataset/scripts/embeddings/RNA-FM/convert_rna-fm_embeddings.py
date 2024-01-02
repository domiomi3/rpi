import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_parquet('/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/results/unique_RNAs.parquet', engine='pyarrow')
print(df.columns)
df = df.sort_values(by='Sequence_1_ID_Unique')
embeds = []
for row in tqdm(df.iterrows()):
    filename = f"/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/RNA-FM/results/{row[1]['Sequence_1_ID_Unique']}.npy"
    arr = np.load(filename)
    padded_seq = np.zeros((150, 640))
    padded_seq[:arr.shape[0], :] = arr
    embeds.append(padded_seq)
    # Pad with zeros

all_embeds = np.array(embeds)
print(all_embeds.shape)
np.save("/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/RNA-FM/rna_embeddings.npy", all_embeds)
