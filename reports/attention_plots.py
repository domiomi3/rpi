from src.models.model import ModelWrapper, RNAProteinInterAct
from src.data.dataloader import get_dataloader
import numpy as np
import matplotlib.pyplot as plt

"""
NOTE: Analysing the interaction plots could give interesting insights to interaction points of given RNA-Protein pairs
"""

def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps.shape[-1])
    attn_maps = attn_maps
    num_heads = 1
    num_layers = 1
    seq_len_1 = attn_maps.shape[0]
    seq_len_2 = attn_maps.shape[1]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps, origin='lower', vmin=0)
            # ax[row][column].set_xticks(list(range(seq_len_1)))
            # ax[row][column].set_xticklabels(list(range(seq_len_1)))
            # ax[row][column].set_yticks(list(range(seq_len_2)))
            # ax[row][column].set_yticklabels(list(range(seq_len_2)))
            # ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
    fig.subplots_adjust(hspace=0.5)
    plt.show()

rna_p_model = RNAProteinInterAct(
    batch_first=True,
   d_model=640,
   num_encoder_layers=1,
   nhead=2,
   dim_feedforward=640,
   )

model = ModelWrapper.load_from_checkpoint("RNAProteinInterAct/8g1iqgp6/checkpoints/epoch=14-step=16428.ckpt", map_location='cpu', model=rna_p_model)

model.eval()

_, val_loader = get_dataloader("Pandas",
                               db_file_train="../dataset/final_train_set.parquet",
                               db_file_valid="../dataset/final_valid_set.parquet",
                               rna_embeddings_path="../dataset/rna_embeddings",
                               protein_embeddings_path="../dataset/protein_embeddings",
                               batch_size=16
                               )

activation = {}
def get_weights(name):
    def hook(model, input, output):
        activation[name] = output
    return hook
rna_embed, protein_embed, y = next(iter(val_loader))

model.model.encoder.layers[0].self_attn_1.register_forward_hook(get_weights('map'))

model.forward(rna_embed, protein_embed)


attn_map = activation['map'][1][8].detach().numpy()

plot_attention_maps(None, attn_map)

print(activation)
