# LCS Discovery and Visualization Environment (LCS-DIVE)

## Installation
LCS-DIVE is written in Python 3. First, you need to download this repository to local. To run, you will also need to first install the LCS-DIVE Python package.

```sh
git clone https://github.com/UrbsLab/LCS-Visualization-Pipeline
cd LCS-Visualization-Pipeline
pip install -r requirements.txt
```

There are 5 files that are runnable from the command line: **AnalysisPhase1.py**, **AnalysisPhase1_pretrained.py**, **AnalysisPhase1_fromstreamline.py**, **AnalysisPhase2.py**, and **NetworkVisualization.py** in the LCSDIVE Folder.
You can run them all from the LCSDIVE folder.

## AnalysisPhase1.py
This file runs ExSTraCS training on your dataset, and is the first file to run on a new dataset. If you have already completed ExSTraCS training from some other pipeline, you should use **AnalysisPhase1_pretrained.py** or **AnalysisPhase1_fromstreamline.py** instead. There exists a few command line arguments:

| Argument | Description | Default |
| ---------- | --------------------  | ---------- |
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
```sh
python AnalysisPhase1.py --d ../Datasets/demodata.csv --o ../Outputs --e lcs --class Class --inst InstanceID --iter 200000 --N 500 --nu 10
```

## AnalysisPhase1_pretrained.py
This file can be run instead of **AnalysisPhase1.py** if you already have presplit train/test datasets AND pretrained ExStraCS models. This file will reorganize those precreated files into a format consistent with the outputs of **AnalysisPhase1.py**, such that AnalysisPhase2.py can be run smoothly. There exists a few command line arguments:

| Argument | Description | Default |
| ---------- | --------------------  | ---------- |
| --d | file path to your directory containing presplit train/test datasets ending with **\_CV_Test/Train.csv** (e.g., dataset1_CV_Test.csv) | MANDATORY |
| --m | file path to your model directory ontaining pretrained ExSTraCS Models labeled ExStraCS_CV (e.g., ExStraCS_0) | MANDATORY |
| --o | file path to your output directory, where LCS-DIVE output files will be directed | MANDATORY |
| --e | experiment name (anything alphanumeric) | MANDATORY |
| --class | column name in dataset of outcome variable | Class |
| --inst | column name in dataset of row id (leave out if N/A) | None |
| --cv | number of CV partitions during training | 3 |
| --random-state | random seed for fixed results | None |
| --cluster | if should run LCS-DIVE Phase 1 on compute cluster | 1 |
| --m1 | cluster job soft memory limit (Gb) | 2 |
| --m2 | cluster job hard memory limit (Gb) | 3 |

Most of these arguments can be left alone in most cases, except for **--d**, **--m**, **--o**, **--e**. Also, make sure **--class**, **--inst**, **--cv** are consistent with your datasets. Additionally, running on the computer cluster will not work until you configure the file's submitClusterJob method to run on your own cluster (it is currently run from a UPenn cluster). Hence, you will need to either configure the method, or set **--cluster** to 0 (runs it locally in serial). Running this LCS-DIVE file on a compute cluster is not very necessary, as it is very quick anyways.

**Sample Run Command**:
```sh
python AnalysisPhase1_pretrained.py --d ../Datasets --m ../Models --o ../Outputs --e test --inst InstanceID --cluster 0
```

## AnalysisPhase1_fromstreamline.py
This file can be run instead of **AnalysisPhase1.py** if you already have run ExStraCS model through STREAMLINE. This file will reorganize those precreated files into a format consistent with the outputs of **AnalysisPhase1.py**, such that AnalysisPhase2.py can be run smoothly. There exists a few command line arguments:

| Argument | Description | Default |
| --s | file path to your STREAMLINE output directory | MANDATORY |
| --e | name of STREAMLINE experiment | MANDATORY |
| --d | name of STREAMLINE dataset to run LCS-DIVE | MANDATORY |
| --o | file path to your LCS-DIVE output directory | MANDATORY |
| --cluster | if should run LCS-DIVE Phase 1 on compute cluster | 1 |
| --m1 | cluster job soft memory limit (Gb) | 2 |
| --m2 | cluster job hard memory limit (Gb) | 3 |

Most of these arguments can be left alone in most cases, except for **--s**, **--e**, **--d**. Additionally, running on the computer cluster will not work until you configure the file's submitClusterJob method to run on your own cluster (it is currently run from a UPenn cluster). Hence, you will need to either configure the method, or set **--cluster** to 0 (runs it locally in serial). Running this LCS-DIVE file on a compute cluster is not very necessary, as it is very quick anyways.

**Sample Run Command**:
```sh
python AnalysisPhase1_fromstreamline.py --s /home/bandheyh/STREAMLINE/lcs/ --e lcs --d demodata --o ../Outputs/ --cluster 0
```

## AnalysisPhase2.py
This file runs the visualization step of LCS-DIVE (Feature Tracking Visualization, Rule Population Visualization, Network Visualization). For a given experiment, this must be run after **AnalysisPhase1.py** or **AnalysisPhase1_pretrained.py**. There exists a few command line arguments:

| Argument | Description | Default |
| ---------- | --------------------  | ---------- |
| --o | for a given experiment, must match that from Phase 1 | MANDATORY |
| --e | for a given experiment, must match that from Phase 1 | MANDATORY |
| --rheight | height to width ratio of rule population heatmaps | 1 |
| --aheight | height to width ratio of feature tracking heatmaps | 1 |
| --cluster | if should run LCS-DIVE Phase 2 on compute cluster | 1 |
| --am1 | feature tracking cluster job soft memory limit (Gb) | 2 |
| --am2 | feature tracking cluster job hard memory limit (Gb) | 3 |
| --rm1 | rule population cluster job soft memory limit (Gb) | 5 |
| --rm2 | rule population cluster job hard memory limit (Gb) | 6 |
| --nm1 | network cluster job soft memory limit (Gb) | 2 |
| --nm2 | network cluster job hard memory limit (Gb) | 3 |
| --dorule | do rule population visualization (sometimes it is too compute expensive to run) | 1 |

Most of these arguments can be left alone in most cases, except for **--o**, **--e**. Additionally, running on the computer cluster will not work until you configure the file's submitClusterJob method to run on your own cluster (it is currently run from a UPenn cluster). Hence, you will need to either configure the method, or set **--cluster** to 0 (runs it locally in serial). Running LCS-DIVE on a compute cluster is highly recommended, as it will speed things up significantly.

By the end of the this Phase, LCS-DIVE has completed running.

**Sample Run Command**:
```sh
python AnalysisPhase2.py --o ../Outputs --e lcs
```

## NetworkVisualization.py
The default network diagram generated by LCS-DIVE for a given dataset is not always exactly how you want it to look. This file provides a GUI interface for you to drag nodes around, resize elements, and resave. For a given experiment, this must be run after **AnalysisPhase2.py**. There exists a few command line arguments:


| Argument | Description | Default |
| ---------- | --------------------  | ---------- |
| --o | for a given experiment, must match that from Phase 1 and 2 | MANDATORY |
| --e | for a given experiment, must match that from Phase 1 and 2| MANDATORY |
| --nodep | Node Power: node relative size is based on a power function occurence^node_power. The larger this is, the faster less relevant nodes vanish | 3 |
| --edgep | Edge Power: edge thickness is based on a power function cooccurence^edge_power. The larger this is, the faster less relevant edges vanish | 3 |
| --nodes | Node Size: node maximum size | 50 |
| --edges | Edge Size: edge maximum thickness | 30 |
| --labelshow | show all node names by default (this being off is useful if there are many features) | 0 |
| --labels | Label Size: node label maximum size | 50 |
| --from_save | Try to open from a previously saved config file | 1 |

**Sample Run Command**:
```sh
python NetworkVisualization.py --o ../Outputs --e lcs
```

Once this command is run, a GUI window will pop up. From there you can
1) Drag nodes around
2) Press **l** over a node to make its label appear/disappear
3) Press X to close the window. This automatically saves your configuration and creates a new visualization. When you run this again, your previously saved configuration will pop up for you to continue working (unless **--from_save** is 0).

