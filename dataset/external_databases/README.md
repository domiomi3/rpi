# Get external databases
## RNAInter v4
```
wget http://www.rnainter.org/raidMedia/download/Download_data_RP.tar.gz
tar -xf Download_data_RP.tar.gz
rm Download_data_RP.tar.gz
```
## NONCODE
```
cd external_databases/NONCODE
wget http://www.noncode.org/datadownload/ncrna_NONCODE[v5.0].fasta.zip
unzip ncrna_NONCODE[v5.0].fasta.zip
rm ncrna_NONCODE[v5.0].fasta.zip
```

## miRBase
```
cd external_databases/miRNA
wget https://www.mirbase.org/ftp/CURRENT/miRNA.dat.gz
tar -xvzf miRNA.dat.gz
```

# Get scraped sequence data
available on KI-SLURM Cluster
```
\data\datasets\RNAProteinInteractions
```