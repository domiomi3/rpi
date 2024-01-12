import argparse
import subprocess
import sys
import os
import shutil

import pandas as pd

from pathlib import Path
from multiprocessing import cpu_count
from tqdm import tqdm
from multiprocessing import Pool
from time import time

src_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_dir))
from utils import row2fasta


def check_and_download_rfam(rfam_dir, cmpress_path):
    """
    Check if the Rfam database is saved locally, if not, download it.
    """
    if not os.path.exists(rfam_dir):
        os.makedirs(rfam_dir)
        print("Rfam directory created.")

    rfam_cm_path = Path(rfam_dir) / "Rfam.cm"
    
    if not rfam_cm_path.exists():
        print("Rfam.cm not found, downloading Rfam files...")
        download_rfam(rfam_dir, cmpress_path)
    else:
        print("Rfam.cm file already exists.")


def download_rfam(rfam_dir, cmpress_path):
    """
    Download the Rfam database and prepare it for use with Infernal.
    """
    subprocess.call(["wget", "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.clanin"], cwd=rfam_dir)
    subprocess.call(["wget", "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz"], cwd=rfam_dir)
    subprocess.call(["gunzip", "Rfam.cm.gz"], cwd=rfam_dir)
    subprocess.call([cmpress_path, "Rfam.cm"], cwd=rfam_dir)


def call_cmscan(family_dir, rfam_cm_path, rfam_clanin_path, cmscan_path, row):
    """
    Scan a single RNA sequence against the Rfam database using Infernal's cmscan.

    Args:
    - family_dir (str): Family directory path.
    - rfam_cm_path (str): Path to the Rfam.cm file.
    - rfam_clanin_path (str): Path to the Rfam.clanin file.
    - cmscan_path (str): Path to the cmscan executable.
    - row (pd.Series): DataFrame row with sequence information.
    
    Returns:
    - None.
    """
    idx = 1 # RNA sequence
    fasta_path = os.path.join(family_dir, f"{row['Id']}.fasta")
    out_path = os.path.join(family_dir, f"infernal_{row['Id']}.tbl")
    row2fasta(row, fasta_path, idx)
   
    subprocess.call([cmscan_path,
                     "--rfam",  # set heuristic filters at Rfam-level (fast)
                     "--cut_ga",
                     "--nohmmonly",  # never run HMM-only mode, not even for models with 0 basepairs
                     "--oskip",  # w/'--fmt 2' and '--tblout', do not output lower scoring overlaps
                     "--tblout", out_path,
                     "--fmt", "2",
                     "--clanin", rfam_clanin_path,
                     "--cpu", str(cpu_count()),
                     rfam_cm_path,
                     fasta_path], stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    
    os.remove(fasta_path)


def parse_family_info(tblout_path):
    """
    Parse the tabular output file from Infernal's cmscan to obtain RNA family 
    information.

    Args:
    - tblout_path (Union[str, Path]): Path to the tabular file output by Infernal's cmscan.

    Returns:
    - pd.DataFrame: DataFrame with parsed RNA family information.
    """
    family_info = []

    with open(tblout_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
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
            'Sequence_1_family': target_name,
        }
        family_info.append(hit_info)

    return pd.DataFrame(family_info)


def assign_family_info(rna_path, rfam_dir, rfam_cm_path, 
                       rfam_clanin_path, cmscan_path):
    """
    Iterate over all available RNA sequences, scan them against the Rfam database
    and add family information to the DataFrame.

    Args:
    - rna_path (str): Path to the RNA sequences file.
    - rfam_dir (str): Path to the Rfam directory.
    - rfam_cm (str): Path to the Rfam.cm file.
    - rfam_clanin (str): Path to the Rfam.clanin file.
    - cmscan_path (str): Path to the cmscan executable.

    Returns:
    - None
    """
    
    # Create temporary subdir or storing .tbl files
    infernal_dir = os.path.join(rfam_dir, "infernal/")
    if not os.path.exists(infernal_dir):
        os.makedirs(infernal_dir)
        print("Infernal directory created.")
    

    # Read RNA sequences and create "Id" column for identyfing hits in rfam database
    df = pd.read_parquet(rna_path, engine='pyarrow')
    df['Id'] = df['Raw_ID1'].astype(str) + '_' + df['Sequence_1_ID'].astype(str)

    start = time()
    
    # Create cmscan arguments for each RNA sequence
    basic_args = [infernal_dir, rfam_cm_path, rfam_clanin_path, cmscan_path]
    args = [basic_args + [row] for _, row in df.iterrows()]

    # Scan all RNA sequences against the Rfam database
    pbar = tqdm(total=len(args), desc="Scanning rfam database")
    jobs = []
    with Pool() as pool:
        for arg_set in args:
            jobs.append(pool.apply_async(call_cmscan, (*arg_set[:4], arg_set[4]), callback=lambda x: pbar.update()))
        pool.close()
        pool.join()
    _ = [job.get() for job in jobs]
    print(f"Elapsed time: {time() - start}")

    # Parse rfam information from .tbl files and save into a DataFrame
    family_info_df = pd.DataFrame()
    skip_cnt = 0
    files_cnt = len(os.listdir(infernal_dir))

    for tbl_file in tqdm(os.listdir(infernal_dir), total=files_cnt, desc="Parsing family info"):
        tbl_path = os.path.join(infernal_dir, tbl_file)
        temp_df = parse_family_info(tbl_path)
        if temp_df.shape[0] > 0:
            family_info_df = pd.concat([temp_df, family_info_df])
        else:
            skip_cnt += 1
        os.remove(tbl_path)

    # Keep only entries with family information and curate the DataFrame
    df = family_info_df.merge(df, on='Id', how='left')
    print(f"Found families for {df.shape[0]}/{files_cnt} sequences.")

    df['Sequence_1_rfam_e_value'] = df['Sequence_1_rfam_e_value'].fillna(1000.0) # set e-value to 1000.0 if no family was found
    df = df.loc[df.groupby(['Id'])['Sequence_1_rfam_e_value'].idxmin()].sort_index() # keep only the best hit (lowest e-val) for each sequence
    df.drop(['Id'], axis=1, inplace=True)
    
    print(f"Number of unique sequences: {df.shape[0]} sequences.")
    # Save DataFrame with family information
    families_path = rna_path[:-8] + "_families.parquet"
    df.to_parquet(families_path, engine='pyarrow')
    print(f"Sequences with families stored at: {families_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scaning available RNA sequences against rfam database.')

    parser.add_argument('--working_dir', type=str, default='/work/dlclarge1/matusd-rpi/RPI', help='Working directory path.')
    parser.add_argument('--results_dir', type=str, default="data/annotations", help='Results directory path.')
    parser.add_argument('--rfam_dir', type=str, default="data/rfam", help='Rfam directory path.')
    parser.add_argument('--rna_short', type=str, default="rna_short.parquet", help='Path to RNA sequences file.')
    parser.add_argument('--rfam_cm', type=str, default="Rfam.cm", help='Path to protein sequences file.')
    parser.add_argument('--rfam_clanin', type=str, default="Rfam.clanin", help='Path to .clanin file.')
    parser.add_argument('--cmscan_path', type=str, default="/home/matusd/.conda/envs/rpi/bin/cmscan", help='Path to CMScan executable.')
    parser.add_argument('--cmpress_path', type=str, default="/home/matusd/.conda/envs/rpi/bin/cmpress", help='Path to CMPress executable.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
     
    args = parser.parse_args()

    working_dir = args.working_dir
    results_dir = args.results_dir
    rfam_dir = args.rfam_dir
    rna_short = args.rna_short
    rfam_cm = args.rfam_cm
    rfam_clanin = args.rfam_clanin
    cmscan_path = args.cmscan_path
    cmpress_path = args.cmpress_path
    seed = args.seed

    os.chdir(working_dir)

    # Curate paths
    rna_path = os.path.join(results_dir, rna_short)
    rfam_cm_path = os.path.join(rfam_dir, rfam_cm)
    rfam_clanin_path = os.path.join(rfam_dir, rfam_clanin)

    check_and_download_rfam(rfam_dir, cmpress_path)
    assign_family_info(rna_path, rfam_dir, rfam_cm_path, rfam_clanin_path, 
                       cmscan_path)
   
    # Remove rfam database
    # shutil.rmtree(rfam_dir)