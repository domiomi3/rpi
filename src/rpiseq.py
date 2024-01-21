# (c) Daniel Fertmann, 2023
# This program insert request on http://pridb.gdcb.iastate.edu/RPISeq/ using selenium
from selenium import webdriver  # pip install selenium
from selenium.webdriver.common.by import By
import pandas as pd


driver = webdriver.Firefox()

test_file = "data/interactions/test_set.parquet"
#inputFile = 'inputData/rpi390_test_set.parquet'
#inputFile = 'inputData/test_set.parquet'

print("loading file " + test_file)

results_file = "output-" + test_file.split("/")[-1] + ".csv"
f = open(results_file, "w")  # Delete existing content
f.write("index\trf\tsvm\n")      # Write header
f.close()
f = open(results_file, "a")


df = pd.read_parquet(test_file) # pip install pyarrow

for index, row in df.iterrows():
    #print(index, row['Sequence_2'], row['Sequence_1'])

    driver.get("http://pridb.gdcb.iastate.edu/RPISeq/")

    protein_seq = row['Sequence_2']
    rna_seq = row['Sequence_1']

    elem = driver.find_element(By.ID, "p_input")
    elem.send_keys(protein_seq)

    elem = driver.find_element(By.ID, "r_input")
    elem.send_keys(rna_seq)

    driver.find_element(By.NAME, "submit").click()

    rf_model = driver.find_element(By.XPATH, "/html/body/div/table/tbody/tr[1]/td[2]/div/table[2]/tbody/tr[4]/td[3]").text
    svm_model = driver.find_element(By.XPATH, "/html/body/div/table/tbody/tr[1]/td[2]/div/table[2]/tbody/tr[5]/td[3]").text

    print("finished " + str(index) + ": rf " + rf_model + " svm " + svm_model)
    f.write(str(index) + "\t" + rf_model + "\t" + svm_model + "\n")

f.close()
print("Finished, output is in file '" + results_file + "'")
driver.close()  # closes the window
