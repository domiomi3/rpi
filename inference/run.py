import csv
from pathlib import Path
import click

"""
NOT FINISHED :(
The idea of this script is to enable an easy inference and evaluation of new datasets. 
A script that takes an database if RNA-protein interactions as an input, 
annotates the db with cluster & RNA family information, creates ready-to-use embeddings, 
create negative interactions and evaluates the annotated database on a trained model. 
This would be helpful to test generalisation on unseen test data.

"""

def check_rna(seq: str):
    unique_letters = set(seq.upper())
    return not unique_letters.difference({'A', 'C', 'G', 'U'})


def check_protein(seq: str):
    unique_letters = set(seq.upper())
    return not unique_letters.difference(
        {'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'})


def check_input(path: Path):
    file = open(path)
    reader = csv.DictReader(file)
    entries = [row for row in reader]
    assert len(entries) != 0, "CSV file should not be empty."
    assert 'protein_seq' in entries[0], "CSV file has no column protein_seq"
    assert 'RNA_seq' in entries[0], "CSV file has no column RNA_seq"
    assert all([check_rna(row['RNA_seq']) for row in entries])
    assert all([check_protein(row['protein_seq']) for row in entries])


def create_protein_embeddings():
    pass

@click.option("--input-csv", default="examples.csv")
@click.option("--esm-model", default="esm2_t6_8M_UR50D.pt")
def main(input_csv):
    check_input(input_csv)



if __name__ == '__main__':
    main()