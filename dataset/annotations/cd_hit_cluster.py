import argparse
import os
import subprocess
import sys
import pandas as pd

from pathlib import Path
from tqdm import tqdm

src_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_dir))
from utils import df2fasta

def run_cd_hit_est(cd_hit_est_path, input_path, output_path, additional_args):
    """
    Runs the CD-HIT-EST tool with specified arguments.

    Args:
    - cd_hit_est_path (str): Path to the CD-HIT-EST executable.
    - input_path (str): Path to the input FASTA file.
    - output_path (str): Path to the output file.
    - additional_args (list): List of additional arguments for CD-HIT-EST.

    Returns:
    - (bool) True if the CD-HIT-EST scan completed successfully, False otherwise.
    """
    command = [cd_hit_est_path, "-i", input_path, "-o", output_path] + additional_args

    try:
        print("Running CD-HIT-EST...")
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)

    except subprocess.CalledProcessError as e:
        print("Error running cd-hit-est:", e)
        return False

    print("CD-HIT-EST completed successfully")
    return True
    

def cluster_sequences(sequence_type, results_dir, sequences_path, cd_hit_est_path):
    """
    Cluster sequences (either protein or RNA) using CD-HIT.

    Args:
    - sequence_type (str): Type of sequences ('proteins' or 'rna').
    - results_dir (str): Directory to store results.
    - sequences_path (str): Path to the parquet file with sequences.
    - cd_hit_est_path (str): Path to the CD-HIT-EST executable.

    Returns:
    - None
    """
    print(f"Clustering {sequence_type} sequences")

    # Create paths for intermediate files
    fasta_path = Path(results_dir, f'inter_{sequence_type}_cluster.fasta')
    ana_path = Path(os.path.join(results_dir, fasta_path.stem + '.ana'))
    ana_clstr_path = Path(os.path.join(results_dir, fasta_path.stem + '.ana.clstr'))

    # Read sequences from the parquet file and convert them to FASTA format
    df = pd.read_parquet(os.path.join(results_dir, sequences_path), engine='pyarrow')
    idx = '2' if sequence_type == 'proteins' else '1'
    df2fasta(df, fasta_path, idx)

    # CD-HIT-EST arguments (remaining are default)):
    # -c: sequence identity threshold, default 0.9
    # -n: word length, default is 10, for longer sequences, it should be smaller
    # -r: 1 or 0, default 1, by default do both +/+ & +/- alignments
 	#     if set to 0, only +/+ strand alignment
    # -M: memory limit (in MB) for the program, default 800; 0 for unlimited;
    # -l: length of throw_away_sequences, default 10
    # -d: length of description in .clstr file, default 20
 	#     if set to 0, it takes the fasta defline and stops at first space
    # -T: number of threads, default 1; with 0, all CPUs will be used

    cd_hit_args = [
        "-c", "0.8", "-n", "5", "-r", "0", "-M", "0", "-l", "5", "-d", "0", 
        "-T", "0"
    ]
    
    if run_cd_hit_est(cd_hit_est_path, fasta_path, ana_path, cd_hit_args):
        # Read the cluster information from the output file
        clusters = []
        with open(ana_clstr_path, 'r') as f:
            lines = f.readlines()  # Read all lines at once

        for line in tqdm(lines, total=len(lines), desc="Processing clusters"):
            line = line.rstrip()
            if 'Cluster' in line:
                current_cluster = line.split()[-1]
            else:
                s_id, score = line.split()[2][1:-3], line.split()[-1]
                clusters.append((current_cluster, s_id, score))

        # Create a DataFrame based on the cluster information (cluster ID, sequence ID, similarity score)
        cluster_col = f'Sequence_{idx}_cluster'
        clusters_df = pd.DataFrame(clusters, columns=[cluster_col, 'ID', f'{cluster_col}_sim'])
        
        print(f"Number of clustered {sequence_type} sequences: {clusters_df.shape[0]}/{df.shape[0]}.")

        # Create a temporary ID column to identify sequences from clusters DataFrame
        df['ID'] = df[f'Raw_ID{idx}'].astype(str) + '_' + df[f'Sequence_{idx}_ID'].astype(str)
        df = df.merge(clusters_df, on='ID')
        df.drop(['ID'], axis=1, inplace=True)
        del clusters_df

        # Create new column indicating if sequence serves as a reference for the cluster
        df[f'{cluster_col}_reference'] = df[f'{cluster_col}_sim'].apply(lambda x: x == '*')

        df.to_parquet(os.path.join(results_dir, f'{sequence_type}_clusters.parquet'), engine='pyarrow')
        
        # Remove intermediate files
        os.remove(fasta_path)
        os.remove(ana_path)
        os.remove(ana_clstr_path)

        print(f"Clustered {sequence_type} sequences stored in {sequence_type}_short_clusters.parquet.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster RNA and protein sequences using CD-HIT.')

    parser.add_argument('--working_dir', type=str, default='/work/dlclarge1/matusd-rpi/RPI', help='Working directory path.')
    parser.add_argument('--results_dir', type=str, default="data/annotations", help='Results directory path.')
    parser.add_argument('--rna_sequences_path', type=str, default="rna_short.parquet", help='Path to RNA sequences file.')
    parser.add_argument('--protein_sequences_path', type=str, default="proteins_short.parquet", help='Path to protein sequences file.')
    parser.add_argument('--cd_hit_est_path', type=str, default="/home/matusd/.conda/envs/rpi/bin/cd-hit-est", help='Path to CD-HIT-EST executable.')
     
    args = parser.parse_args()

    working_dir = args.working_dir
    results_dir = args.results_dir
    rna_sequences_path = args.rna_sequences_path
    protein_sequences_path = args.protein_sequences_path
    cd_hit_est_path = args.cd_hit_est_path

    os.chdir(working_dir)
    
    cluster_sequences('proteins', args.results_dir, args.protein_sequences_path, args.cd_hit_est_path)
    cluster_sequences('rna', args.results_dir, args.rna_sequences_path, args.cd_hit_est_path)
