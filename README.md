# Heterogeneous Treatment Assignment Effect Estimation Under Non-compliance with Conditional Front-door Adjustment

- Run synthetic dataset A experiments: `src/run_batch_main_sim_rate.sh`
- Run synthetic dataset B experiments: `src/run_batch_main_sim_.sh`
- Run IHDP experiments: `./src/run_batch_main_ihdp_rate.sh`
- Run AMR-UTI experiments: `./src/run_batch_main_amruti.sh`
    - Running AMR-UTI experiments requires accessing three data filew from https://clinicalml.org/data/amr-dataset/.
        - `all_prescriptions.csv`
        - `all_uti_features.csv`
        - `all_uti_resist_labels.csv`
    - Once the above data is downloaded, move them into `data` folder for proper script execution.
- Codes for plotting the results are under: `src/notebooks/`
    - Variance visualization results: `src/notebooks/variance_viz.ipynb`
    - Synthetic experiments results: `src/notebooks/sim_experiment_results.ipynb`
    - IHDP experiments results: `src/notebooks/ihdp_experiment_results.ipynb`
    - AMR-UTI experiments results: `src/notebooks/ihdp_experiment_results.ipynb`