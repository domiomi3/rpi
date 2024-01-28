# (c) Daniel Fertmann, 2023
# This program inserts request on http://pridb.gdcb.iastate.edu/RPISeq/ using selenium
import argparse
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from torchmetrics.classification import BinaryPrecision, BinaryRecall, \
    BinaryF1Score, BinaryAccuracy, BinaryAUROC


def scrape_rpiseq(test_file, output_file):
    # Initialize the Chrome driver
    driver = webdriver.Chrome()

    # Load the test data
    df = pd.read_parquet(test_file)

    # Open the file for writing
    with open(output_file, 'w') as file:
        # Write the header
        file.write("index,label,rf_prediction,svm_prediction\n")

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Scraping data from RPISeq"):
            driver.get("http://pridb.gdcb.iastate.edu/RPISeq/")

            protein_seq = row['Sequence_2']
            rna_seq = row['Sequence_1']
            label = row['interaction']

            elem = driver.find_element(By.ID, "p_input")
            elem.send_keys(protein_seq)

            elem = driver.find_element(By.ID, "r_input")
            elem.send_keys(rna_seq)

            driver.find_element(By.NAME, "submit").click()

            rf_model = driver.find_element(By.XPATH, "/html/body/div/table/tbody/tr[1]/td[2]/div/table[2]/tbody/tr[4]/td[3]").text
            svm_model = driver.find_element(By.XPATH, "/html/body/div/table/tbody/tr[1]/td[2]/div/table[2]/tbody/tr[5]/td[3]").text

            # Write the results line by line
            file.write(f"{index},{label},{rf_model},{svm_model}\n")

    # Close the driver
    driver.close()


def calculate_metrics(output_file):
    df = pd.read_parquet(output_file)

    # Ensure the 'label' column is numeric and handle NaNs
    true_labels = pd.to_numeric(df['label'], errors='coerce').dropna().astype(np.float32)
    true_labels_tensor = torch.tensor(true_labels.values)

    # Calculating metrics for RF and SVM
    for model in ['rf_prediction', 'svm_prediction']:
        # Ensure the prediction columns are numeric, handle non-numeric values and NaNs
        predictions = pd.to_numeric(df[model], errors='coerce').fillna(0).astype(np.float32)
        predictions_tensor = torch.tensor(predictions.values)

        # Initialize metric objects
        accuracy = BinaryAccuracy()
        precision = BinaryPrecision()
        recall = BinaryRecall()
        f1 = BinaryF1Score()
        auroc = BinaryAUROC()

        # Update the metric objects
        accuracy.update(predictions_tensor, true_labels_tensor)
        precision.update(predictions_tensor, true_labels_tensor)
        recall.update(predictions_tensor, true_labels_tensor)
        f1.update(predictions_tensor, true_labels_tensor)
        auroc.update(predictions_tensor, true_labels_tensor)

        print(f"Metrics for {model}:")
        print(f"Accuracy: {accuracy.compute()}")
        print(f"Precision: {precision.compute()}")
        print(f"Recall: {recall.compute()}")
        print(f"F1 Score: {f1.compute()}")
        print(f"AUROC: {auroc.compute()}")


def main(args):
    results_file = "rpi_seq-" + args.test_file.split("/")[-1]

    scrape_rpiseq(args.test_file, results_file)

    print("Calculating metrics...")
    calculate_metrics(results_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Scrape RPISeq data and save predictions.')
    parser.add_argument('--test_file', type=str, default="data/interactions/rpi2825_test_set.parquet", help='Path to the test set file.')
    args = parser.parse_args()
    
    main(args)
