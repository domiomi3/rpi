import subprocess
import pandas as pd
import numpy as np

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from pathlib import Path
from tqdm import tqdm
import uuid
from subprocess import PIPE, run
import os
import shutil

"""
Clusters protein sequences with CD-HIT.
Requires https://github.com/weizhongli/cdhit
pip install Biopython
"""

CD_HIT_EST_PATH = "/work/dlclarge1/matusd-rpi/cd-hit/cd-hit-est"

RNA_SEQUENCES_PATH = "dataset/scripts/annotate/dataset/results/rna_sequences_short.parquet"
PROTEIN_SEQUENCES_PATH = "/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/results/protein_sequences_short.parquet"

rng = np.random.default_rng(seed=0)
WORKING_DIR = '/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/shuffle'
Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)
RESULTS_DIR = "/work/dlclarge1/matusd-rpi/RPI/dataset/scripts/annotate/dataset/results"


def df2fasta(df: pd.DataFrame, fasta_path: Path, idx: str):
    with open(fasta_path, 'w') as handle:
        sequences = [SeqRecord(Seq(row[f'Sequence_{idx}']), id=f"{row[f'Raw_ID{idx}']}_{row[f'Sequence_{idx}_ID']}") for
                     _, row in
                     df.iterrows()]
        SeqIO.write(sequences, handle, "fasta")


def get_new_sequences(cluster, row):
    # required to check if a sequence is already in a cluster
    out_path = Path(WORKING_DIR, f"{str(uuid.uuid4())}_cluster_2.fasta")
    shuffled = row['Sequence_1_ID']
    fasta_path_keep = Path(WORKING_DIR, f'{str(uuid.uuid4())}_keep.fasta')
    fasta_path_reduce = Path(WORKING_DIR, f"{str(uuid.uuid4())}_reduced.fasta")

    df2fasta(cluster, fasta_path_reduce, "1")
    with open(fasta_path_keep, 'w') as handle:
        SeqIO.write([SeqRecord(Seq(shuffled), id=row['Id'])], handle, "fasta")
    base_args = [CD_HIT_EST_PATH,
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
    # required to check if a sequence is already in a cluster
    out_path = Path(WORKING_DIR, f"{str(uuid.uuid4())}_cluster_2.fasta")
    shuffled = row['Sequence_2_ID']
    fasta_path_keep = Path(WORKING_DIR, f'{str(uuid.uuid4())}_keep.fasta')
    fasta_path_reduce = Path(WORKING_DIR, f"{str(uuid.uuid4())}_reduced.fasta")

    df2fasta(cluster, fasta_path_reduce, "2")
    with open(fasta_path_keep, 'w') as handle:
        SeqIO.write([SeqRecord(Seq(shuffled), id=row['Id'])], handle, "fasta")
    base_args = [CD_HIT_EST_PATH,
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
    fasta_path = Path(WORKING_DIR, 'inter_protein_cluster.fasta')
    ana_path = Path(os.path.join(WORKING_DIR, fasta_path.stem + '.ana'))
    ana_clstr_path = Path(os.path.join(WORKING_DIR, fasta_path.stem + '.ana.clstr'))
    df = pd.read_parquet(PROTEIN_SEQUENCES_PATH, engine='pyarrow')
    df2fasta(df, fasta_path, "2")

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
    result = run([CD_HIT_EST_PATH, "-i", str(fasta_path.resolve()),
                  "-o", str(ana_path.resolve())] + args, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    print(result.returncode, result.stdout, result.stderr)

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
    df.to_parquet(os.path.join(RESULTS_DIR, 'protein_sequences_clusters.parquet'), engine='pyarrow')
    print("Proteins with clustered stored.")


def cluster_rnas():
    """
    Cluster RNA sequences
    """
    fasta_path = Path(WORKING_DIR, 'inter_rna_cluster.fasta')
    ana_path = Path(os.path.join(WORKING_DIR, fasta_path.stem + '.ana'))
    ana_clstr_path = Path(os.path.join(WORKING_DIR, fasta_path.stem + '.ana.clstr'))

    # df = pd.read_pickle('data/inter_family_benchmark.plk.gz')
    df = pd.read_parquet(RNA_SEQUENCES_PATH, engine='pyarrow')

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
    result = run([CD_HIT_EST_PATH, "-i", str(fasta_path.resolve()),
                  "-o", str(ana_path.resolve())] + args, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    print(result.returncode, result.stdout, result.stderr)

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
    # clusters.to_parquet('clusters.parquet', engine='pyarrow')

    df['Id'] = df['Raw_ID1'].astype(str) + '_' + df['Sequence_1_ID'].astype(str)
    df = df.merge(clusters, on='Id')

    df['Sequence_1_cluster_reference'] = df['Sequence_1_cluster_sim'].apply(lambda x: x == '*')

    df = df.drop(['Id'], axis=1)
    df.to_parquet(os.path.join(RESULTS_DIR, 'rna_sequences_clusters.parquet'), engine='pyarrow')
    print("RNAs with clustered stored.")


if __name__ == '__main__':
    cluster_proteins()
    cluster_rnas()
    shutil.rmtree(WORKING_DIR)
