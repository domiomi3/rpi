import os
import click
import pandas as pd


@click.command()
@click.option('--embedding-path')
@click.option('--rna-path')
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
    check_results()
