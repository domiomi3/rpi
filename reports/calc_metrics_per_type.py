from torchmetrics import MetricCollection

from src.models.model import ModelWrapper, RNAProteinInterAct
from src.data.dataloader import get_dataloader
from torch import nn
from tqdm import tqdm
from statistics import mean
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryAUROC
import click
from collections import defaultdict
import pandas as pd
import json
import torch

loss = nn.BCELoss()

"""
This script calculates the performance of a given & trained model on each interaction type.
As discussed in the master thesis, different distributions of interaction types can lead to different performance on 
different data splits.
"""


def calc_confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    prediction = torch.round(prediction)

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


@click.command()
@click.option("--device", default='cpu')
@click.option("--num-encoder-layers", default=6)
@click.option("--num-dataloader-workers", default=2)
@click.option("--batch-size", default=16)
@click.option("--d-model", default=320)
@click.option("--n-head", default=2)
@click.option("--dim-feedforward", default=640)
@click.option("--dropout", default=0.1)
@click.option("--key-padding-mask", is_flag=True, show_default=True, default=False, help="Enables key padding mask")
@click.option("--lr-init", default=0.001)
@click.option("--weight-decay", default=0.001)
@click.option("--dataloader-type", default="Pandas")
@click.option("--protein-embeddings-path", default="dataset/protein_embeddings")
@click.option("--rna-embeddings-path", default="dataset/rna_embeddings")
@click.option("--db-file-valid", default='dataset/final_valid_set.parquet')
@click.option("--seed", default=0)
@click.option("--checkpoint-path")
def main(device: str,
         num_encoder_layers: int,
         num_dataloader_workers: int, batch_size: int,
         d_model: int, n_head: int, dropout: float,
         dim_feedforward: int, lr_init: float,
         weight_decay: float,
         key_padding_mask: bool, dataloader_type: str,
         protein_embeddings_path: str, rna_embeddings_path: str,
         db_file_valid: str,
         checkpoint_path: str,
         seed: int):
    rna_p_model = RNAProteinInterAct(
        batch_first=True,
        embed_dim=640,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        nhead=n_head,
        dim_feedforward=dim_feedforward,
        key_padding_mask=key_padding_mask,
        norm_first=True,
        dropout=dropout
    )

    model = ModelWrapper.load_from_checkpoint(checkpoint_path,
                                              map_location=device, model=rna_p_model, lr_init=lr_init,
                                              weight_decay=weight_decay, seed=seed)

    val_loader, _ = get_dataloader(dataloader_type,
                                   db_file_train=db_file_valid,
                                   db_file_valid=db_file_valid,
                                   rna_embeddings_path=rna_embeddings_path,
                                   protein_embeddings_path=protein_embeddings_path,
                                   num_workers=num_dataloader_workers,
                                   batch_size=batch_size
                                   )

    model.eval()
    vals = []
    metrics = MetricCollection([
        BinaryPrecision(),
        BinaryRecall(),
        BinaryF1Score(),
        BinaryAccuracy(),
        BinaryAUROC()
    ])
    metrics = metrics.to(device)
    tp, fp, tn, fn = 0, 0, 0, 0
    confusion_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    db = pd.read_parquet(db_file_valid, engine='pyarrow')

    for rna_embed, protein_embed, y, rna_inter_id in tqdm(val_loader):
        rna_embed = rna_embed.to(device)
        protein_embed = protein_embed.to(device)

        pred = model.forward(rna_embed, protein_embed).float()
        y = y.float().to(device)
        pred = pred.reshape(pred.shape[0])
        loss_val = loss(pred, y)
        # metrics.update(pred, y)
        # vals.append(loss_val.detach().item())
        tp_tmp, fp_tmp, tn_tmp, fn_tmp = calc_confusion(pred, y)
        tp += tp_tmp
        fp += fp_tmp
        tn += tn_tmp
        fn += fn_tmp
        if len(rna_inter_id) != 1:
            continue

        res_df = db[db['RNAInterID'] == rna_inter_id[0]][
            ['Category1', 'Category2']]
        assert res_df.shape[0] == 1
        rna_category, protein_category = res_df.values.tolist()[0]
        interaction_type = f"{rna_category}-{protein_category}"
        categories = [
            ('rna-category', rna_category),
            ('protein-category', protein_category),
            ('interaction-type', interaction_type),
        ]
        for cat_name, cat in categories:
            confusion_matrix[cat_name][cat]['tp'] += tp_tmp
            confusion_matrix[cat_name][cat]['fp'] += fp_tmp
            confusion_matrix[cat_name][cat]['tn'] += tn_tmp
            confusion_matrix[cat_name][cat]['fn'] += fn_tmp

    print("Accuracy per categories")
    # calc metrics for each category
    for cat_name, cat_values in confusion_matrix.items():
        for cat_value, m in cat_values.items():
            m['precision'] = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) != 0 else 0
            m['accuracy'] = (m['tp'] + m['tn']) / (m['tp'] + m['tn'] + m['fn'] + m['fp'])
            m['recall'] = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) != 0 else 0
            m['f1'] = (2 * m['precision'] * m['recall']) / (m['precision'] + m['recall']) if (m['precision'] + m[
                'recall']) != 0 else 0
            m['total'] = m['tp'] + m['tn'] + m['fn'] + m['fp']
        assert sum([m['total'] for m in cat_values.values()]) == len(val_loader)
        confusion_matrix[cat_name] = dict(sorted(cat_values.items(), key=lambda x: x[1]['accuracy'], reverse=True))
    for cat_name, cat_values in confusion_matrix.items():
        print("\t", cat_name)
        for cat_value, m in cat_values.items():
            print(
                f"\t\t{cat_value:<15s}: {round(m['accuracy'] * 100, 2)}: {round(m['total'] / len(val_loader) * 100, 2)}")
    print('#' * 20)

    # store results
    results = [
        {
            "interaction-type": cat_value,
            "f1": m['f1'],
            "accuracy": m['accuracy'],
            "total_num": m['total']
        } for cat_value, m in confusion_matrix['interaction-type'].items()]
    with open('final_results/test-set-metrics-per-interaction-type.json', 'w') as f:
        json.dump(results, f)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (2 * precision * recall) / (precision + recall)
    single_class_metrics = [
        (precision, "precision"),
        (recall, "recall"),
        (f_measure, "F1-Score")
    ]
    print(f"Calc single class metric for {db_file_valid}")
    for metric, metric_name in single_class_metrics:
        print(f"{metric_name:<25s}{metric}")

    print(f"Calc metric for {db_file_valid}")
    metric_dict = metrics.compute()
    for key, value in metric_dict.items():
        print(f"{key:<25s}{value}")
    print(f"{'Loss':<25s}t{mean(vals)}")


if __name__ == '__main__':
    main()
