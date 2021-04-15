# LCS Discovery and Visualization Environment (LCS-DIVE)

## Installing and Running LCS-DIVE Files
LCS-DIVE is written in Python 3. First, you need to download this repository to local. To run, you will also need to first install the LCS-DIVE Python package.

'''sh
pip install LCS-DIVE
'''

There are 4 files that are runnable from the command line: **AnalysisPhase1.py**, **AnalysisPhase1_pretrainedJob.py**, **AnalysisPhase2.py**, and **NetworkVisualization.py**.

### AnalysisPhase1.py
This file runs ExSTraCS training on your dataset, and is the first file to run on a new dataset. If you have already completed ExSTraCS training from some other pipeline, you should use **AnalysisPhase1_pretrainedJob.py** instead. There exists a few command line arguments:

| Argument | Description | Default |
| --d | file path to your dataset (can be txt, csv, or gz) | MANDATORY |
| --o | file path to your output directory, where LCS-DIVE output files will be directed | MANDATORY |
| --e | experiment name (anything alphanumeric) | MANDATORY |
| --class | column name in dataset of outcome variable | Class |
| --inst | column name in dataset of row id (leave out if N/A) | None |
| --group | column name in dataset of group id (leave out if N/A) | None |
| --match | column name in dataset of match id for matched CV (leave out if N/A) | None |
| --cv | number of CV partitions during training | 3 |
| --iter | number of ExSTraCS learning iterations | 16000 |
| --N | maximum ExSTraCS micropopulation | 1000 |
| --nu | ExStraCS hyperparameter | 1000 |
| --at-method | feature tracking method | wh |
| --rc | rule compaction method | None |
| --random-state | random seed for fixed results | None |
| --fssample | skrebate feature selection sample size | 1000 |
| --cluster | if should run LCS-DIVE Phase 1 on compute cluster | 1 |
| --m1 | cluster job soft memory limit (Gb) | 2 |
| --m2 | cluster job hard memory limit (Gb) | 3 |

Most of these arguments can be left alone in most cases, except for **--d**, **--o**, **--e**. Additionally, running on the computer cluster will not work until you configure the file's submitClusterJob method to run on your own cluster (it is currently run from a UPenn cluster). Hence, you will need to either configure the method, or set **--cluster** to 0 (runs it locally in serial). Running LCS-DIVE on a compute cluster is highly recommended, as it will speed things up significantly.

**Sample Run Command**:
'''sh
python AnalysisPhase1.py --d ../Datasets/dataset1.csv --o ../Outputs --e dataset1_test1 --inst Customer_ID --iter 200000 --N 500 --nu 10 --cluster 0
'''


