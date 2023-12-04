import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy, BinaryAUROC
from typing import Callable

"""
Calculates defined metrics (Accuracy, Recall, Precision, F1Score, AUROC) 
for a "fake" dataset with 2 negative and 1 positive datapoint 
and for training and validation dataset
with a random classifier, positive classifier and negative classifier
"""

class Classifier:
    def __init__(self, name: str, classifier_func: Callable):
        self.name = name
        self.classifier_func = classifier_func


class RNAInterActionsPandasFake(torch.utils.data.Dataset):
    def __init__(self, db_file):
        self.db = pd.read_parquet(db_file, engine='pyarrow')
        self.db = self.db.assign(row_number=range(len(self.db)))
        self.length = self.db.shape[0]

    def __len__(self) -> int:
        """length special method"""
        # return self.n_users()
        return self.length

    def __getitem__(self, index):
        result_df = self.db[self.db['row_number'] == index][
            ['Sequence_1_shuffle', 'Sequence_2_shuffle']]
        assert result_df.shape[0] == 1
        seq_1_shuffle, seq_2_shuffle = result_df.values.tolist()[0]
        interacts = float(not seq_1_shuffle and not seq_2_shuffle)
        return interacts


def main():
    train_set = RNAInterActionsPandasFake("../dataset/dataset_v4/final_train_set_reduced.parquet")
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    valid_set = RNAInterActionsPandasFake("../dataset/dataset_v4/final_test_set_reduced.parquet")
    valid_loader = DataLoader(valid_set, batch_size=512, shuffle=False)

    metrics = MetricCollection([
        BinaryAccuracy(),
        BinaryRecall(),
        BinaryPrecision(),
        BinaryF1Score(),
        BinaryAUROC()
    ])
    target = torch.IntTensor([1, 0, 0])
    print("Metrics for fake dataset:")
    classifiers = [
        Classifier(name="Only positives", classifier_func=get_ones),
        Classifier(name="Only negatives", classifier_func=get_zeros),
        Classifier(name="Random", classifier_func=get_random)
    ]
    for classifier in classifiers:
        print(f"\tClassifier: {classifier.name}")
        for _ in range(50_000):
            y_hat = classifier.classifier_func(target.shape[0])
            metrics(y_hat, target)
        metrics_dict = metrics.compute()
        print_metrics(metrics_dict)
        metrics.reset()

    dataloaders = [("train", train_loader), ("valid", valid_loader)]
    for dataloader in dataloaders:
        print(f"Dataset: {dataloader[0]}")
        for classifier in classifiers:
            print(f"\tClassifier: {classifier.name}")
            for y in dataloader[1]:
                y_hat = classifier.classifier_func(y.shape[0])
                metrics(y_hat, y)
            metrics_dict = metrics.compute()
            print_metrics(metrics_dict)
            metrics.reset()


def get_ones(size: int):
    return torch.IntTensor([1 for _ in range(size)])


def get_zeros(size: int):
    return torch.IntTensor([0 for _ in range(size)])


def get_random(size: int):
    return torch.IntTensor([random.choice([0, 1]) for _ in range(size)])


def print_metrics(metrics: dict, prefix="\t\t"):
    for metric, value in metrics.items():
        print(f"{prefix}{metric}: {round(float(value), 4)}")


if __name__ == '__main__':
    main()
