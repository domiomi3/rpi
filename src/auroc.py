import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import interp
from torchmetrics import AUROC, ROC

def calculate_metrics(file_path):
    df = pd.read_parquet(file_path)
    preds = torch.tensor(df['logits'].values, dtype=torch.float32)
    target = torch.tensor(df['labels'].values, dtype=torch.int32)
    
    # AUROC calculation
    auroc = AUROC(task="binary")
    auroc_score = auroc(preds, target).item()

    # ROC calculation (using torchmetrics, which provides FPR and TPR values)
    roc = ROC(task="binary")
    fpr, tpr, thresholds = roc(preds, target)

    return fpr.numpy(), tpr.numpy(), auroc_score

def main(experiment_files, output_file):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif" 
    plt.rcParams["font.size"] = 10  # Sets global font size

    plt.figure(figsize=(4, 4))
    mean_fpr = np.linspace(0, 1, 100)
    
    for exp, files in experiment_files.items():
        tprs = []
        auroc_scores = []
        for file in files:
            fpr, tpr, auroc_score = calculate_metrics(file)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            auroc_scores.append(auroc_score)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(auroc_scores)
        std_auc = np.std(auroc_scores)

        plt.plot(mean_fpr, mean_tpr, label=f'{exp} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)
        
        print(f"{exp}: Mean AUC = {mean_auc:.2f}, Std AUC = {std_auc:.2f}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right", fontsize=8, title_fontsize=8)  # Set legend font size to 8
    plt.grid(True)

    # Adjust tick sizes
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()  # Adjust layout to make room for the axis labels

    plt.savefig(output_file)
    print(f"ROC curve plot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and Plot ROC and AUROC using torchmetrics")
    parser.add_argument('--experiment2', nargs='+', required=True, help="Paths to the Parquet files for Experiment 2")
    parser.add_argument('--experiment3', nargs='+', required=True, help="Paths to the Parquet files for Experiment 3")
    parser.add_argument('--experiment1', nargs='+', required=True, help="Paths to the Parquet files for Experiment 4")
    parser.add_argument('--output', type=str, default='roc_curves.pdf', help="Output file path for the ROC curve plot")
    
    args = parser.parse_args()

    experiment_files = {
        'RPIembeddor': args.experiment1,
        'xRPI': args.experiment2,
        'IPMiner': args.experiment3
    }

    main(experiment_files, args.output)
