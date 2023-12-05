"""
You need to instll Infernal

either via anaconda:

  conda install -c bioconda infernal

or via:

  wget 'eddylab.org/infernal/infernal-1.1.3-linux-intel-gcc.tar.gz' && tar -xvzf infernal-*.tar.gz && rm infernal-*.tar.gz

"""

import subprocess

import pandas as pd

from typing import Union
from pathlib import Path
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
from multiprocessing import cpu_count
import os
from tqdm import tqdm
from multiprocessing import Pool
from time import time

# needs to be downloaded
# e.g. download_rfam_cms function
CM_PATH = 'family/Rfam.cm'
CLANIN_PATH = 'family/Rfam.clanin'

CM_SCAN_PATH = "cmscan"
RNA_SEQUENCES_PATH = '../../results/rna_sequences_short.parquet'
WORKING_DIR = "family"
Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)

def df2fasta(df: pd.DataFrame, fasta_path: Path):
    with open(fasta_path, 'w') as handle:
        sequences = [SeqRecord(Seq(row['Sequence_1']), id=f"{row['Raw_ID1']}_{row['Sequence_1_ID']}") for _, row in
                     df.iterrows()]
        SeqIO.write(sequences, handle, "fasta")


def row2fasta(row, fasta_path: Path):
    with open(fasta_path, 'w') as handle:
        sequence = [SeqRecord(Seq(row['Sequence_1']), id=f"{row['Raw_ID1']}_{row['Sequence_1_ID']}")]
        SeqIO.write(sequence, handle, "fasta")


class InfernalTbloutParser():
    """
    Parse the output of a tabular file resulting from applying Infernal to search
    the rfam database to get family information for a given set of query sequences.


    """

    def __init__(self,
                 tblout_path: Union[str, Path],
                 ):
        self._tbl_path = tblout_path

    def parse(self):
        family_info = []

        with open(self._tbl_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith('#'):
                continue
            line = [l for l in line.split() if l]

            idx, target_name, t_accession, query_name, q_accession, clan_name, mdl, \
                mdl_from, mdl_to, seq_from, seq_to, strand, trunc, pass_, gc, bias, \
                score, e_value, inc, olp, anyidx, afrct1, afrct2, winidx, wfrct1, \
                wfrct2 = line[:26]

            description = ' '.join(line[26:])

            hit_info = {
                'Id': query_name,
                'Sequence_1_rfam_q_accession': q_accession,
                'Sequence_1_family': target_name,
                'Sequence_1_rfam_t_accession': t_accession,
                'Sequence_1_rfam_description': description,
                'Sequence_1_rfam_e_value': float(e_value),
            }

            family_info.append(hit_info)

        return pd.DataFrame(family_info)


def download_rfam_cms(destination):
    subprocess.call(["wget", "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.clanin"], cwd=destination)
    subprocess.call(["wget", "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz"], cwd=destination)
    subprocess.call(["gunzip", "Rfam.cm.gz"], cwd=destination)
    subprocess.call(["/Users/lars/Downloads/infernal/src/cmpress", "Rfam.cm"], cwd=destination)
    # you can do that here or later, better here
    # probably... jsut don't do it twice cause command fails


# Only run once!
# download_rfam_cms("family")


def call_cmscan(row):
    fasta_path = os.path.join(WORKING_DIR, f"{row['Id']}.fasta")
    out_path = os.path.join(WORKING_DIR, f"infernal_{row['Id']}.tbl")
    row2fasta(row, fasta_path)
    subprocess.call([CM_SCAN_PATH,
                     "--rfam",  # set heuristic filters at Rfam-level (fast)
                     "--cut_ga",
                     "--nohmmonly",  # never run HMM-only mode, not even for models with 0 basepairs
                     "--oskip",  # w/'--fmt 2' and '--tblout', do not output lower scoring overlaps
                     "--tblout", out_path,
                     "--fmt", "2",
                     "--clanin", CLANIN_PATH,
                     "--cpu", str(cpu_count()),
                     CM_PATH,
                     fasta_path], stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    os.remove(fasta_path)


def rfam_scan_single_sequence(row: pd.Series, family: str):
    # define row
    row['Id'] = row['Raw_ID1'] + '_' + row['Sequence_1_ID']
    df = row.rename(None).to_frame()
    out_path = os.path.join(WORKING_DIR, f"infernal_{row['Id']}.fasta")
    call_cmscan(row)

    parser = InfernalTbloutParser(tblout_path=out_path)
    family_info = parser.parse()
    os.remove(out_path)
    if family_info.shape[0] == 0:
        return False
    print("Found some family!!!")
    family_info = family_info.merge(df.T, on='Id', how='outer')

    family_info['Sequence_1_rfam_e_value'] = family_info['Sequence_1_rfam_e_value'].fillna(1000.0)
    family_info = family_info.fillna('unknown')

    # returns the corresponding sequences according to the family with the lowest e-value
    family_info = family_info.loc[family_info.groupby(['Id'])['Sequence_1_rfam_e_value'].idxmin()].sort_index()
    assert family_info.shape[0] == 1
    return family_info.iloc[0]['Sequence_1_family'] == family


def main():
    df = pd.read_parquet(RNA_SEQUENCES_PATH, engine='pyarrow')
    df['Id'] = df['Raw_ID1'].astype(str) + '_' + df['Sequence_1_ID'].astype(str)

    start = time()
    args = [row for _, row in df.iterrows()]
    pbar = tqdm(total=len(args))
    jobs = []
    with Pool() as pool:
        for arg_set in args:
            jobs.append(pool.apply_async(call_cmscan, (arg_set,), callback=lambda x: pbar.update()))
        pool.close()
        pool.join()
    _ = [job.get() for job in jobs]
    print(f"Elapsed time with Pool: {time() - start}")
    # Both of the following commands use infernal
    # run cmpress if you run pipeline for first time... or run it after install of stuff....
    # subprocess.call(["cmpress", cm_path])
    family_info = pd.DataFrame()
    for path in os.listdir(WORKING_DIR):
        if not path.endswith('.tbl'):
            continue
        out_path = os.path.join(WORKING_DIR, path)
        parser = InfernalTbloutParser(tblout_path=out_path)
        temp_df = parser.parse()
        family_info = pd.concat([temp_df, family_info])
        os.remove(out_path)

    family_info = family_info.merge(df, on='Id', how='outer')

    family_info['Sequence_1_rfam_e_value'] = family_info['Sequence_1_rfam_e_value'].fillna(1000.0)
    family_info = family_info.fillna('unknown')

    # returns the corresponding sequences according to the family with the lowest e-value
    family_info = family_info.loc[family_info.groupby(['Id'])['Sequence_1_rfam_e_value'].idxmin()].sort_index()
    family_path = RNA_SEQUENCES_PATH[:-8] + "_families.parquet"
    family_info.to_parquet(family_path, engine='pyarrow')
    print(f"Sequences with families stored at: {family_path}")


if __name__ == '__main__':
    main()
