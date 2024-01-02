import os
import click
import pandas as pd
import argparse

def check_results(embedding_path: str, rna_path: str):
    result_files = [f for f in os.listdir(embedding_path) if os.path.isfile(os.path.join(embedding_path, f)) if f.endswith('.npy')]
    embeddings_df = pd.DataFrame([dict(
        Sequence_1_ID_Unique=x[:-4]
    ) for x in result_files])

    rna_df = pd.read_parquet(rna_path)
    rna_df = rna_df[['Sequence_1_ID_Unique']].drop_duplicates()
    # should be fulfilled
    print(f"Number of Unique RNAs: {rna_df.shape[0]}")
    print(f"Number of Embeddings: {embeddings_df.shape[0]}")
    assert rna_df.shape[0] == embeddings_df.shape[0], "Number of Proteins is unequal to Number of Embeddings"
    # difference should not exist
    set_1 = set(rna_df['Sequence_1_ID_Unique'].astype(str).unique())
    set_2 = set(embeddings_df['Sequence_1_ID_Unique'].astype(str).unique())

    assert len(set_1 - set_2) == 0
    assert len(set_2 - set_1) == 0
    print("Everything is fine :)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding_path', type=str, default="/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/embeddings/RNA-FM/results/", help='Path to folder with RNA embeddings')
    parser.add_argument('--rna_path', type=str, default='/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/results/unique_RNAs.parquet', help='Path to RNA parquet file')

    args = parser.parse_args()
    check_results(args.embedding_path, args.rna_path)
