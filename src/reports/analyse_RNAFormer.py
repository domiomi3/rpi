from statistics import mean, stdev
import json
from collections import defaultdict

def process_files(files: list) -> defaultdict[list]:
    results = defaultdict(list)
    for file in files:
        with open(file) as f:
            result = json.load(f)
            for k, v in result[0].items():
                results[k].append(v)
    return results

def print_metrics(dataset_name: str, results: defaultdict[list]):
    print(f"Dataset: {dataset_name}")
    for metric, values in results.items():
        print(f"\t{metric}: \t {round(mean(values), 3)}+-{round(stdev(values), 3)}")

random_test_split_files = [
    "rna-former/rna-former-random-test-split-0-47.json",
    "rna-former/rna-former-random-test-split-1-47.json",
    "rna-former/rna-former-random-test-split-2-47.json",
]

test_split_files = [
    "rna-former/rna-former-test-split-0-47.json",
    "rna-former/rna-former-test-split-1-47.json",
    "rna-former/rna-former-test-split-2-47.json",
]

rpi_split_files = [
    "rna-former/rna-former-rpi2825-0-47.json",
    "rna-former/rna-former-rpi2825-1-47.json",
    "rna-former/rna-former-rpi2825-2-47.json",
]

random_test_split_results = process_files(random_test_split_files)
test_split_results = process_files(test_split_files)
rpi2825_results = process_files(rpi_split_files)

print_metrics("Random Test Split", random_test_split_results)
print_metrics("Test Split", test_split_results)
print_metrics("RPI2825", rpi2825_results)