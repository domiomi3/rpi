import subprocess
import pandas as pd
import numpy as np

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from pathlib import Path
from tqdm import tqdm
import uuid

import os

"""
Clusters protein sequences with CD-HIT.
Requires https://github.com/weizhongli/cdhit
"""

CD_HIT_PATH = "cd-hit"
CD_HIT_EST_PATH = "cd-hit-est"

rng = np.random.default_rng(seed=0)
working_dir = '../shuffle'


def df2fasta(df: pd.DataFrame, fasta_path: Path, idx: str):
    with open(fasta_path, 'w') as handle:
        sequences = [SeqRecord(Seq(row[f'Sequence_{idx}']), id=f"{row[f'Raw_ID{idx}']}_{row[f'Sequence_{idx}_ID']}") for _, row in
                     df.iterrows()]
        SeqIO.write(sequences, handle, "fasta")


def get_new_sequences(cluster, row):
    out_path = Path(working_dir, f"{str(uuid.uuid4())}_cluster_2.fasta")
    shuffled = row['Sequence_1_ID']
    fasta_path_keep = Path(working_dir, f'{str(uuid.uuid4())}_keep.fasta')
    fasta_path_reduce = Path(working_dir, f"{str(uuid.uuid4())}_reduced.fasta")

    df2fasta(cluster, fasta_path_reduce, "1")
    with open(fasta_path_keep, 'w') as handle:
        SeqIO.write([SeqRecord(Seq(shuffled), id=row['Id'])], handle, "fasta")
    base_args = [CD_HIT_PATH,
                 '-i', str(fasta_path_keep.resolve()),
                 '-i2', str(fasta_path_reduce.resolve()),
                 '-o', str(out_path.resolve())]

    std_args = ['-c', str(0.8), '-T', '0', '-n', str(5), '-s', str(0.0), '-s2', str(0.0), '-g', '0', '-r', '0']

    args = base_args + std_args

    p = subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    file = open(out_path, 'r')
    keep = [record for record in SeqIO.parse(file, "fasta")]
    file.close()
    # del fasta_path_keep
    os.remove(fasta_path_keep)
    os.remove(out_path)
    os.remove(fasta_path_reduce)
    # del out_path

    if len(keep) == len(cluster):
        # print('### no hits')
        return False
    else:
        print('### some hit')
        return True


def get_new_sequences_protein(cluster, row):
    out_path = Path(working_dir, f"{str(uuid.uuid4())}_cluster_2.fasta")
    shuffled = row['Sequence_2_ID']
    fasta_path_keep = Path(working_dir, f'{str(uuid.uuid4())}_keep.fasta')
    fasta_path_reduce = Path(working_dir, f"{str(uuid.uuid4())}_reduced.fasta")

    df2fasta(cluster, fasta_path_reduce, "2")
    with open(fasta_path_keep, 'w') as handle:
        SeqIO.write([SeqRecord(Seq(shuffled), id=row['Id'])], handle, "fasta")
    base_args = [CD_HIT_PATH,
                 '-i', str(fasta_path_keep.resolve()),
                 '-i2', str(fasta_path_reduce.resolve()),
                 '-o', str(out_path.resolve())]

    std_args = ['-c', str(0.8), '-T', '0', '-n', str(5), '-s', str(0.0), '-s2', str(0.0), '-g', '0']

    args = base_args + std_args

    p = subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    file = open(out_path, 'r')
    keep = [record for record in SeqIO.parse(file, "fasta")]
    file.close()
    # del fasta_path_keep
    os.remove(fasta_path_keep)
    os.remove(out_path)
    os.remove(fasta_path_reduce)
    # del out_path

    if len(keep) == len(cluster):
        # print('### no hits')
        return False
    else:
        print('### some hit')
        return True


def cluster_proteins():
    fasta_path = Path(working_dir, 'inter_protein_cluster.fasta')
    ana_path = Path(fasta_path.stem + '.ana')
    ana_clstr_path = Path(ana_path.stem + '.ana.clstr')
    df = pd.read_parquet('../../results/rpi2825/protein_sequences.parquet', engine='pyarrow')
    df2fasta(df, fasta_path, "2")

    args = [
        "-c", str(0.9),
        "-n", str(5),
        "-aS", str(0.0),
        "-s", str(0.0),
        "-g", "0",
        "-r", "0",
        "-M", "0",
        "-l", str(5),
        "-d", "0",
        "-T", "0",
    ]

    subprocess.call([CD_HIT_PATH, "-i", str(fasta_path.resolve()),
                     "-o", str(ana_path.resolve())] + args, stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    clusters = []
    with open(ana_clstr_path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            # print(line)
            if 'Cluster' in line:
                current_cluster = line.split()[-1]
            else:
                s_id = line.split()[2][1:-3]
                score = line.split()[-1]
                clusters.append((current_cluster, s_id, score))
    clusters = pd.DataFrame(clusters, columns=['Sequence_2_cluster', 'Id', 'Sequence_2_cluster_sim'])

    df['Id'] = df['Raw_ID2'].astype(str) + '_' + df['Sequence_2_ID'].astype(str)
    df = df.merge(clusters, on='Id')

    df['Sequence_2_cluster_reference'] = df['Sequence_2_cluster_sim'].apply(lambda x: x == '*')
    df = df.drop(['Id'], axis=1)
    df.to_parquet('../results/rpi2825/protein_sequences_clusters.parquet', engine='pyarrow')
    print(df.columns)
    print("Proteins with clustered stored.")


def main():
    """
    Cluster RNA sequences
    """
    fasta_path = Path(working_dir, 'inter_family_test.fasta')
    ana_path = Path(fasta_path.stem + '.ana')
    ana_clstr_path = Path(ana_path.stem + '.ana.clstr')

    # df = pd.read_pickle('data/inter_family_benchmark.plk.gz')
    df = pd.read_parquet('../../results/rpi2825/rna_sequences.parquet', engine='pyarrow')

    df2fasta(df, fasta_path, "1")

    args = [
        "-c", str(0.8),
        "-n", str(5),
        "-aS", str(0.0),
        "-s", str(0.0),
        "-g", "0",
        "-r", "0",
        "-M", "0",
        "-l", str(5),
        "-d", "0",
        "-T", "0",
    ]
    subprocess.call([CD_HIT_EST_PATH, "-i", str(fasta_path.resolve()),
                     "-o", str(ana_path.resolve())] + args, stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT)  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    clusters = []
    with open(ana_clstr_path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            # print(line)
            if 'Cluster' in line:
                current_cluster = line.split()[-1]
            else:
                s_id = line.split()[2][1:-3]
                score = line.split()[-1]
                clusters.append((current_cluster, s_id, score))
    clusters = pd.DataFrame(clusters, columns=['Sequence_1_cluster', 'Id', 'Sequence_1_cluster_sim'])
    clusters.to_parquet('clusters.parquet', engine='pyarrow')

    df['Id'] = df['Raw_ID1'].astype(str) + '_' + df['Sequence_1_ID'].astype(str)
    df = df.merge(clusters, on='Id')

    df['Sequence_1_cluster_reference'] = df['Sequence_1_cluster_sim'].apply(lambda x: x == '*')

    df = df.drop(['Id'], axis=1)
    df.to_parquet('rna_sequences_clusters.parquet', engine='pyarrow')
    print("RNAs with clustered stored.")

    args = []
    for cluster_id, cluster in tqdm(df.groupby('Sequence_1_cluster')):
        fasta_path_reduce = Path(working_dir, f'{cluster_id}_reduce.fasta')
        df2fasta(cluster, fasta_path_reduce, "1")
        for i, row in cluster.iterrows():
            args.append((cluster, row, fasta_path_reduce))

    """
    pbar = tqdm(total=len(args))
    jobs = []
    with Pool() as pool:
        for arg_set in args:
            jobs.append(pool.apply_async(get_new_sequences, arg_set, callback=lambda x: pbar.update()))
        pool.close()
        pool.join()
    results = [job.get() for job in jobs]
    """
    # df = df.drop(['Id'], axis=1)
    df.to_parquet('../results/rpi2825/rna_sequences_clusters.parquet', engine='pyarrow')
    print(df.columns)
    print("RNAs with clustered stored.")
    # results = pool.starmap(get_new_sequences, args)


if __name__ == '__main__':
    cluster_proteins()