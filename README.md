# 🦞LobsterNet: Heterogeneous Treatment Assignment Effect Estimation Under Non-adherance

## 🔍 Overview
- **LobsterNet** is a multitask neural network designed for heterogeneous treatment assignment effect estimation under treatment non-adherence.

- It leverages the **conditional front-door adjustment (CFD)**, which theoretically gaurantees lower variance estimates than the commonly used standard backdoor adjustment (CFD) when the true treatment effect is small.

- This repository contains experimental results in "**Heterogeneous Treatment Assignment Effect Estimation Under Non-adherance with Conditional Front-door Adjustment**" (CHIL 2025).

---
## ▶️ Quick Start

### 1: obtain required datasets
- Synthetic datasets will be automatically generated in the following steps
- IHDP dataset is provided under `data/IHDP/ihdp.RData`, originally downloaded from the [npci repository](https://github.com/vdorie/npci/blob/master/examples/ihdp_sim/data/ihdp.RData)
- AMR-UTI dataset need to be obtained from [PhysioNet](https://physionet.org/content/antimicrobial-resistance-uti/1.0.0/), and place the `all_prescriptions.csv`, `all_uti_features.csv`, and `all_uti_resist_labels.csv` files in `data/AMR-UTI` folder.

### 2: Install required packages
```bash
    pip install -r requirements.txt
```

### 3: train and evaluate SBD and CFD estimators:
| Path                              | Description                                |
|-----------------------------------|--------------------------------------------|
| `src/run_batch_main_sim_A.sh`   | Run synthetic dataset A experiments method |
| `src/run_batch_main_sim_B.sh`   | Run synthetic dataset B experiments usage  |
| `src/run_batch_main_ihdp.sh`    | Run IHDP experiments                       |
| `src/run_batch_main_amruti.sh`  | Run AMR-UTI experiments                    |

### 4: analyze results
| Path                                          | Description                                    |
|-----------------------------------------------|------------------------------------------------|
| `src/analysis/variance_viz.ipynb`             | Analyze asymptotic variance                    |
| `src/analysis/sim_experiment_results.ipynb`   | Analyze synthetic datasets experiment results  |
| `src/analysis/ihdp_experiment_results.ipynb`  | Analyze IHDP experiments results               |
| `src/analysis/amruti_experiment_results.ipynb`| AMR-UTI experiments results                    |


---

## 📝 Citation

If you use LobsterNet in your research, please cite:

> Chen W, Chang T, Wiens J. *Conditional Front-door Adjustment for Heterogeneous Treatment Assignment Effect Estimation Under Non-adherence.* *Conference on Health, Inference and Learning* (2025).

---

## 🛠️ License

This project is licensed under the [Apache 2.0 License](LICENSE).