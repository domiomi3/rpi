import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_parquet('unique_RNAs.parquet', engine='pyarrow')
print(df.columns)
df = df.sort_values(by='Sequence_1_ID_Unique')
embeds = []
for row in tqdm(df.iterrows(), total=len(df)):
    filename = f"rna_embeddings_rna_former/{row[1]['Sequence_1_ID_Unique']}.npy"
    arr = np.load(filename)
    padded_seq_1_embed = np.zeros((150, 150, 256))
    padded_seq_1_embed[:arr.shape[0], :arr.shape[1], :] = arr
    embeds.append(padded_seq_1_embed)
    # Pad with zeros

all_embeds = np.array(embeds)
print(all_embeds.shape)
np.save("rna_embeddings_rna_former.npy", all_embeds)