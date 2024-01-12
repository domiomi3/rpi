import argparse
import subprocess
import os
import sys
import subprocess

import pandas as pd

from pathlib import Path
from tqdm import tqdm
from time import time
from multiprocessing import Pool


src_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_dir))
from utils import row2fasta

def download_pfam(pfam_dir, hmmpress_path):
    """
    Download and extract Pfam database and PfamScan tool.

    Args:
    - pfam_dir (str): Directory to download the files.
    - hmmpress_path (str): Path to the hmmpress executable.

    Returns:
    - None
    """
    # Download and extract the PfamScan tool
    pfam_scan_dir = os.path.join(pfam_dir, "PfamScan")
    if not os.path.exists(pfam_scan_dir):
        subprocess.call(["wget", "https://ftp.ebi.ac.uk/pub/databases/Pfam/Tools/PfamScan.tar.gz"], cwd=pfam_dir)
        subprocess.call(["tar", "-xzvf", "PfamScan.tar.gz"], cwd=pfam_dir)
        os.remove(os.path.join(pfam_dir, "PfamScan.tar.gz"))
    else:
        print("PfamScan directory already exists.")

    # Download and extract the Pfam database
    if not os.path.exists(os.path.join(pfam_dir, "Pfam-A.hmm")):
        subprocess.call(["wget", "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"], cwd=pfam_dir)
        subprocess.call(["gunzip", "Pfam-A.hmm.gz"], cwd=pfam_dir)
    else:
        print("Pfam-A.hmm file already exists.")
    if not os.path.exists(os.path.join(pfam_dir, "Pfam-A.hmm.dat")):
        subprocess.call(["wget", "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz"], cwd=pfam_dir)
        subprocess.call(["gunzip", "Pfam-A.hmm.dat.gz"], cwd=pfam_dir)
    else:
        print("Pfam-A.hmm.dat file already exists.")
    if not os.path.exists(os.path.join(pfam_dir, "active_site.dat")):
        subprocess.call(["wget", "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/active_site.dat.gz"], cwd=pfam_dir)
        subprocess.call(["gunzip", "active_site.dat.gz"], cwd=pfam_dir)

    # Generate binary files for Pfam-A.hmm 
    subprocess.call([hmmpress_path, "Pfam-A.hmm"], cwd=pfam_dir)


def download_hmmer(pfam_dir): 
    hmmer_scan_dir = os.path.join(pfam_dir, "hmmer-3.4")
    install_path = "/work/dlclarge1/matusd-rpi/"

    if not os.path.exists(hmmer_scan_dir):
        # Following instructions on http://hmmer.org/documentation.html
        subprocess.call(["wget", "http://eddylab.org/software/hmmer/hmmer.tar.gz"], cwd=pfam_dir)
        subprocess.call(["tar", "-xzvf", "hmmer.tar.gz"], cwd=pfam_dir)
        os.remove(os.path.join(pfam_dir, "hmmer.tar.gz"))

        os.chdir(os.path.join(pfam_dir, "hmmer-3.4"))

        # Create binaries in the install_path
        subprocess.run(["./configure", f"--prefix={install_path}"], check=True)
        subprocess.run(["make"], check=True)
        subprocess.run(["make", "install"], check=True)

    else:
        print("hmmer directory already exists.")

    # Add the binaries to the path
    os.environ["PATH"] += f":{install_path}/hmmer3"


def call_pfamscan(pfam_dir, pfam_results_dir, pfam_scan_path, row):
    """
    Scan a single protein sequence against the Pfam database using PfamScan.

    Args:
    - pfam_dir (str): Directory with all required Pfam files.
    - pfam_results_dir (str): Directory storing .fast and .tbl files.
    - pfam_scan_path (str): Path to the pfam_scan.pl file.
    - row (pd.Series): DataFrame row with sequence information.
    
    Returns:
    - None
    """
    # breakpoint()
    idx = 2 # protein sequence
    fasta_path = os.path.join(pfam_results_dir, f"{row[f'Raw_ID{idx}']}.fasta")
    out_path = os.path.join(pfam_results_dir, f"{row[f'Raw_ID{idx}']}.tbl")
    row2fasta(row, fasta_path, idx)

    subprocess.run([pfam_scan_path,"-fasta", fasta_path, "-dir", pfam_dir,
                     "-outfile", out_path], 
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    os.remove(fasta_path)

def assign_clan_info(protein_path, pfam_dir, pfam_scan_path):
    """
    Iterate over all available protein sequences, scan them against the Pfam database
    and add clan information to the DataFrame.

    Args:
    - protein_path (str): Path to the protein sequences file.
    - pfam_dir (str): Path to the Pfam directory.
    - pfam_scan_path (str): Path to the pfam_scan.pl file.
    Returns:
    - None
    """
        # Create temporary subdir or storing .tbl files
    pfam_results_dir = os.path.join(pfam_dir, "results")
    if not os.path.exists(pfam_results_dir):
        os.makedirs(pfam_results_dir, exist_ok=True)
    
    # Read RNA sequences and create "Id" column for identyfing hits in rfam database
    df = pd.read_parquet(protein_path, engine='pyarrow')

    start = time()
    # Create pfam_scan arguments for each protein sequence
    basic_args = [pfam_dir, pfam_results_dir, pfam_scan_path]
    args = [basic_args + [row] for _, row in df.iterrows()]

    # Scan all RNA sequences against the Rfam database
    pbar = tqdm(total=len(args), desc="Scanning pfam database")
    jobs = []
    with Pool() as pool:
        for arg_set in args:
            jobs.append(pool.apply_async(call_pfamscan, (*arg_set[:3], arg_set[3]), callback=lambda x: pbar.update()))
        pool.close()
        pool.join()
    _ = [job.get() for job in jobs]
    print(f"Elapsed time: {time() - start}")

    #Parse pfam information from .tbl files and save into a DataFrame
    clan_info_df = pd.DataFrame()
    skip_cnt = 0
    files_cnt = len(os.listdir(pfam_results_dir))

    # breakpoint()
    for tbl_file in tqdm(os.listdir(pfam_results_dir), total=files_cnt, desc="Parsing pfam info"):
        tbl_path = os.path.join(pfam_results_dir, tbl_file)
        temp_df = parse_tbl_file(tbl_path)
        # breakpoint()
        if temp_df.shape[0] > 0:
            clan_info_df = pd.concat([temp_df, clan_info_df])
        else:
            skip_cnt += 1
        os.remove(tbl_path)

    # Merge with the original DataFrame
    # breakpoint()
    df = clan_info_df.merge(df, on='Raw_ID2', how='left')

    print(f"\nFound clans for {df.shape[0]}/{files_cnt} sequences.")
    print(f"Number of unique clans: {df['Sequence_2_clan'].nunique()}.")

    # Save DataFrame with family information
    clan_path = protein_path[:-8] + "_clans.parquet"
    df.to_parquet(clan_path, engine='pyarrow')
    print(f"Sequences with clans stored at: {clan_path}")


def parse_tbl_file(file_path):
    """
    Parse pfam search infromation from a .tbl file.

    Args:
    - file_path (str): Path to the .tbl file.

    Returns:
    - clan_info_df (pd.DataFrame): DataFrame with clan and Raw_ID2 information.
    """

    parsed_data = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue  # Skip comment or empty lines

            seq_id, al_start, al_end, env_start, env_end, hmm_acc, hmm_name, \
            type, hmm_start, hmm_end, hmm_length, bit_score, e_val, significance, \
            clan = line.split()
            if clan == 'No_clan':
                continue  # Skip entries with 'No_clan'

            if seq_id not in parsed_data or parsed_data[seq_id]['e_value'] > e_val:
                parsed_data[seq_id] = {'e_value': e_val, 'clan': clan}

    clan_info_list = [{'Raw_ID2': seq_id.split('_')[0], 'Sequence_2_clan': data['clan']} for seq_id, data in parsed_data.items()]
    return pd.DataFrame(clan_info_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scaning available RNA sequences against rfam database.')

    parser.add_argument('--working_dir', type=str, default='/work/dlclarge1/matusd-rpi/RPI', help='Working directory path.')
    parser.add_argument('--annotations_dir', type=str, default="data/annotations", help='Results directory path.')
    parser.add_argument('--pfam_dir', type=str, default="data/pfam/", help='Pfam directory path.')
    parser.add_argument('--protein_short', type=str, default="proteins_short.parquet", help='Path to protein sequences file.')
    parser.add_argument('--hmmpress_path', type=str, default="/work/dlclarge1/matusd-rpi/bin/hmmpress", help='Path to hmmpress executable.')
    parser.add_argument('--pfam_scan_path', type=str, default="data/pfam/PfamScan/pfam_scan.pl", help='Path to pfam_scan.pl')
     
    args = parser.parse_args()

    working_dir = args.working_dir
    annotations_dir = args.annotations_dir
    pfam_dir = args.pfam_dir
    protein_short = args.protein_short
    hmmpress_path = args.hmmpress_path
    pfam_scan_path = args.pfam_scan_path

    os.chdir(working_dir)

     # Curate paths
    protein_path = os.path.join(annotations_dir, protein_short)


    download_hmmer(pfam_dir)
    download_pfam(pfam_dir, hmmpress_path)
    assign_clan_info(protein_path, pfam_dir, pfam_scan_path)
