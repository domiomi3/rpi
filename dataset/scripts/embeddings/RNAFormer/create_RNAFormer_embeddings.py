import os
import tqdm
import argparse
import pathlib
import torch
import urllib.request
import logging

import torch.cuda
import numpy as np
import pandas as pd

from RNAformer.model.RNAformer import RiboFormer
from RNAformer.pl_module.datamodule_rna import IGNORE_INDEX, PAD_INDEX
from RNAformer.utils.data.rna import CollatorRNA
from RNAformer.utils.configuration import Config
from eval_predictions import evaluate, print_dict_tables

"""
This script is a fork from the infer_riboformer.py script
https://github.com/automl/RNAformer/blob/main/infer_riboformer.py
"""

logger = logging.getLogger(__name__)


class EvalRNAformer():

    def __init__(self, model_dir, precision=16, flash_attn=False):

        model_dir = pathlib.Path(model_dir)

        config = Config(config_file=model_dir / 'config.yml')
        state_dict = torch.load(model_dir / 'state_dict.pth', map_location='cpu')

        if precision == 32 or flash_attn == False:
            config.trainer.precision = 32
            config.RNAformer.precision = 32
            config.RNAformer.flash_attn = False
        elif precision == 16 or precision == 'fp16':
            config.trainer.precision = 16
            config.RNAformer.precision = 16
            config.RNAformer.flash_attn = True
        elif precision == 'bf16':
            config.trainer.precision = 'bf16'
            config.RNAformer.precision = 'bf16'
            config.RNAformer.flash_attn = True

        model_config = config.RNAformer
        model_config.seq_vocab_size = 5
        model_config.max_len = state_dict["seq2mat_embed.src_embed_1.embed_pair_pos.weight"].shape[1]

        model = RiboFormer(model_config)
        model.load_state_dict(state_dict, strict=True)

        # model = model.cuda()
        if precision == 16 or precision == 'fp16' or precision == 'bf16':
            model = model.half()
        self.model = model.eval()

        self.ignore_index = IGNORE_INDEX
        self.pad_index = PAD_INDEX

        self.collator = CollatorRNA(self.pad_index, self.ignore_index)

    def __call__(self, sequence: str, mean_triual=True):
        length = len(sequence)
        sequence = sequence.upper()
        sequence = sequence.replace('X', '')
        sequence = sequence.replace('T', '')
        sequence = sequence.replace('I', '')
        seq_vocab = ['A', 'C', 'G', 'U', 'N']
        seq_stoi = dict(zip(seq_vocab, range(len(seq_vocab))))
        int_sequence = list(map(seq_stoi.get, sequence))

        input_sample = torch.LongTensor(int_sequence)

        input_sample = {'src_seq': input_sample, 'length': torch.LongTensor([len(input_sample)])[0]}
        batch = self.collator([input_sample])
        with torch.no_grad():
            # logits, mask = self.model(batch['src_seq'].cuda(), batch['length'].cuda(), infer_mean=True)
            return self.model(batch['src_seq'], batch['length'], infer_mean=True)
        sample_logits = logits[0, :length, :length, -1].detach()
        # triangle mask
        if mean_triual:
            low_tr = torch.tril(sample_logits, diagonal=-1)
            upp_tr = torch.triu(sample_logits, diagonal=1)
            mean_logits = (low_tr.t() + upp_tr) / 2
            sample_logits = mean_logits + mean_logits.t()

        pred_mat = torch.sigmoid(sample_logits) > 0.5

        return pred_mat.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RNAformer')
    parser.add_argument('-n', '--model_name', type=str, default="ts0_conform_dim256_32bit")
    parser.add_argument('-m', '--model_dir', type=str, )
    parser.add_argument("-r", "--rna_file", type=str, default="unique_RNAs.parquet")
    parser.add_argument('-f', '--flash_attn', type=bool, default=False )
    parser.add_argument('-p', '--precision', type=int, default=32 )
    parser.add_argument('-s', '--save_predictions', type=bool, default=False )

    args, unknown_args = parser.parse_known_args()

    if args.model_dir is None:
        model_dir = f"checkpoints/{args.model_name}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            print("Downloading model checkpoints")
            urllib.request.urlretrieve(
                f"https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/{args.model_name}/config.yml",
                f"checkpoints/{args.model_name}/config.yml")
            urllib.request.urlretrieve(
                f"https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/{args.model_name}/state_dict.pth",
                f"checkpoints/{args.model_name}/state_dict.pth")
    else:
        model_dir = args.model_dir


    eval_model = EvalRNAformer(model_dir, precision=args.precision, flash_attn=args.flash_attn)

    def count_parameters(parameters):
        return sum(p.numel() for p in parameters)
    print(f"Model size: {count_parameters(eval_model.model.parameters())}")

    file = args.rna_file
    df = pd.read_parquet(file)

    processed_samples = []
    for idx, sample in tqdm.tqdm(enumerate(df.to_dict('records')), total=df.shape[0]):
        sequence = sample['Sequence_1']
        pred_mat = eval_model(sequence, mean_triual=True)
        pred_mat = pred_mat.squeeze(0)
        pred_mat = pred_mat.detach()
        np.save(f"data/rpi2825/embeddings/{sample['Sequence_1_ID_Unique']}.npy", pred_mat)
