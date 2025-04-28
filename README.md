# Heterogeneous Treatment Assignment Effect Estimation Under non-adherance with Conditional Front-door Adjustment

This is the official code repository for reproducing experimental results in "Heterogeneous Treatment Assignment Effect Estimation Under Non-adherance with Conditional Front-door Adjustment" (CHIL 2025).

### Step 1: obtain required datasets:
- Synthetic datasets will be automatically generated in the following steps
- IHDP dataset is provided under `data/IHDP/ihdp.RData`, originally downloaded from the [npci repository](https://github.com/vdorie/npci/blob/master/examples/ihdp_sim/data/ihdp.RData)
- AMR-UTI dataset need to be obtained from [PhysioNet](https://physionet.org/content/antimicrobial-resistance-uti/1.0.0/), and place the `all_prescriptions.csv`, `all_uti_features.csv`, and `all_uti_resist_labels.csv` files in `data/AMR-UTI` folder.


### Step 2: train and evaluate SBD and CFD estimators:
- Run synthetic dataset A experiments: `./src/run_batch_main_sim_A.sh`
- Run synthetic dataset B experiments: `./src/run_batch_main_sim_B.sh`
- Run IHDP experiments: `./src/run_batch_main_ihdp.sh`
- Run AMR-UTI experiments: `./src/run_batch_main_amruti.sh`

### Step 3: analyzing results:
- Analyze asymptotic variance: `src/notebooks/variance_viz.ipynb`
- Analyze synthetic datasets experiment results: `src/notebooks/sim_experiment_results.ipynb`
- IHDP experiments results: `src/notebooks/ihdp_experiment_results.ipynb`
- AMR-UTI experiments results: `src/notebooks/amruti_experiment_results.ipynb`