import os, argparse, random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from data.simulation import generate_synthetic_nc_rate
from utils import rmse, plot_continuous_estimation
from estimators.standard import mlp_inference, tlearner_inference, train_tlearner_grid_search
from estimators.propensity import dragon_inference, train_dragon_grid_search
from estimators.frontdoor import cfd_inference, train_lobster_grid_search, lobster_inference, train_cfd_tlearner_grid_search


def main(args):
    ### Initialization ###
    random.seed(args.rep)
    np.random.seed(args.rep)
    tf.random.set_seed(args.rep)
    rpath = os.path.join("Synthetic-rate", f"n={args.n}_p={args.p}_amp={args.amplitude}", f"Uay={args.u_ay}_Uty={args.u_ty}",
        f"type={args.non_compliance_type}", f"rate={args.non_compliance_rate}", f"rep={args.rep}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, rpath, args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    figure_dir = os.path.join(args.figure_dir, rpath, args.model)
    os.makedirs(figure_dir, exist_ok=True)

    ### Load & preprocess data ###
    train_df, test_df = generate_synthetic_nc_rate(
        n=args.n, p=args.p, rep=args.rep, nc_rate=args.non_compliance_rate, amplitude=args.amplitude,
        nc_type=args.non_compliance_type, num_u_ty=args.u_ty, num_u_ay=args.u_ay
    )
    X_cols = [c for c in train_df.columns if c.startswith("X")]
    X_train, X_test = train_df[X_cols].values, test_df[X_cols].values
    T_train, T_test = train_df["T"].values.astype(int), test_df["T"].values.astype(int)
    A_train, A_test = train_df["A"].values.astype(int), test_df["A"].values.astype(int)
    Y_train, Y_test = train_df["Y"].values, test_df["Y"].values
    
    At_train = train_df[["A_T0", "A_T1"]].values
    At_test = test_df[["A_T0", "A_T1"]].values
    Ya_train = train_df[["Y_A0", "Y_A1"]].values
    Ya_test = test_df[["Y_A0", "Y_A1"]].values
    Yt_train = train_df[["Y_T0", "Y_T1"]].values
    Yt_test = test_df[["Y_T0", "Y_T1"]].values

    ### Train model ###
    epochs=1000
    early_stop=True
    po_act="sigmoid"
    batch_size_list=[128]
    reg_l2_list=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 0.0]
    learning_rate_list=[1e-4]
    is_sbd = "sbd" in args.model
    if is_sbd:
        if args.model == "sbd-tlearner":
            model, metric_dict, hparam_dict = train_tlearner_grid_search(X_train, T_train, Y_train, checkpoint_dir=checkpoint_dir, 
                overwrite=args.overwrite, batch_size_list=batch_size_list, verbose=False, early_stop=early_stop, 
                reg_l2_list=reg_l2_list, learning_rate_list=learning_rate_list, epochs=epochs, po_act=po_act)
            Yt_pred_test = tlearner_inference(model, X_test, batch_size=hparam_dict["batch_size"])
        elif args.model == "sbd-dragon":
            model, metric_dict, hparam_dict = train_dragon_grid_search(X_train, T_train, Y_train, checkpoint_dir=checkpoint_dir, 
                overwrite=args.overwrite, batch_size_list=batch_size_list, verbose=True, epochs=epochs, po_act=po_act,
                early_stop=early_stop, reg_l2_list=reg_l2_list, learning_rate_list=learning_rate_list)
            Yt_pred_test, _ = dragon_inference(model, X_test, batch_size=hparam_dict["batch_size"])
    else:
        if args.model == "cfd-tlearner":
            model_t, model_a, model_y_t0, model_y_t1, metric_dict, hparam_dict = train_cfd_tlearner_grid_search(
                X_train, A_train, T_train, Y_train, early_stop=early_stop, verbose=True, po_act=po_act,
                checkpoint_dir=checkpoint_dir, overwrite=args.overwrite, batch_size_list=batch_size_list, 
                reg_l2_list=reg_l2_list, learning_rate_list=learning_rate_list, epochs=epochs, shared_backbone=False)
            t_pred_test = mlp_inference(model_t, X_test, batch_size=hparam_dict["best_t_model_hparam"]["batch_size"])
            a_pred_test = tlearner_inference(model_a, X_test, batch_size=hparam_dict["best_a_model_hparam"]["batch_size"])
            y_pred_t0_test = tlearner_inference(model_y_t0, X_test, batch_size=hparam_dict["best_y_t0_model_hparam"]["batch_size"])
            y_pred_t1_test = tlearner_inference(model_y_t1, X_test, batch_size=hparam_dict["best_y_t1_model_hparam"]["batch_size"])
            y_pred_test = np.hstack((y_pred_t0_test, y_pred_t1_test))
        elif args.model == "lobster":
            model, metric_dict, hparam_dict = train_lobster_grid_search(
                X_train, A_train, T_train, Y_train, epochs=epochs, po_act=po_act,
                checkpoint_dir=checkpoint_dir, overwrite=args.overwrite, early_stop=early_stop, 
                shared_backbone=True, shared_outcome=True, verbose=True, 
                batch_size_list=batch_size_list, reg_l2_list=reg_l2_list, learning_rate_list=learning_rate_list, 
                alpha=max(1.0, np.sqrt(np.square(Y_train).mean())), beta=max(1.0, np.sqrt(np.square(Y_train).mean())))
            t_pred_test, a_pred_test, y_pred_test = lobster_inference(model, X_test, batch_size=hparam_dict["batch_size"])
        Ya_pred_test, Yt_pred_test = cfd_inference(t_pred_test, a_pred_test, y_pred_test, 
                                                   non_compliance_type=args.non_compliance_type)
        
    ### Plot loss function ###
    if is_sbd:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        axes = [ax]
    else: fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if args.model == "sbd-dragon":
        axes[0].plot(metric_dict["val_loss"], label="overall loss")
        axes[0].plot(metric_dict["val_outcome_loss"], label="outcome loss")
        axes[0].plot(metric_dict["val_treatment_loss"], label="treatment loss")
    elif args.model.startswith("sbd-"):
        axes[0].plot(metric_dict["val_loss"], label="overall loss")
    elif args.model.startswith("lobster"):
        axes[0].plot(metric_dict["val_outcome_loss"], label="outcome loss")
        axes[0].plot(metric_dict["val_loss"], label="overall loss")
    else:
        axes[0].plot(metric_dict["val_t0_outcome_loss"], label="t0 outcome loss")
        axes[0].plot(metric_dict["val_t1_outcome_loss"], label="t1 outcome loss")
    if not is_sbd:
        axes[0].plot(metric_dict["val_compliance_loss"], label="compliance loss")
        axes[0].plot(metric_dict["val_treatment_loss"], label="treatment loss")
        axes[0].legend()
        axes[1].plot(metric_dict["val_compliance_accuracy"], label="compliance accuracy")
        axes[1].plot(metric_dict["val_treatment_accuracy"], label="treatment accuracy")
        axes[1].legend()
        axes[1].legend()
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "loss.png"))

    ### Evaluate estimation ###
    eval_dict = {"value": [], "label": []}
    yt_pehe = rmse((Yt_pred_test[:, 1]-Yt_pred_test[:, 0]), (Yt_test[:, 1]-Yt_test[:, 0]))
    yt_rpehe = yt_pehe / np.sqrt(np.square(Yt_test[:, 1]-Yt_test[:, 0]).mean())
    print(f"pehe: {yt_pehe:.3f}, r-pehe: {yt_rpehe:.3f}")
    eval_dict["value"].append(yt_pehe)
    eval_dict["label"].append("assign effect PEHE")
    eval_dict["value"].append(yt_rpehe)
    eval_dict["label"].append("assign effect relative PEHE")

    eval_dict = pd.DataFrame(eval_dict)
    eval_dict.to_csv(os.path.join(checkpoint_dir, "eval.csv"), index=False)

    full_stage_fig_path = os.path.join(figure_dir, f"potential_outcome_estimates.png")
    fig = plot_continuous_estimation(Yt_pred_test, Yt_test, title="Potential treatment assignment estimation")
    fig.savefig(full_stage_fig_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="path to data directory", default="../data")
    parser.add_argument('--figure_dir', type=str, help="path to data directory", default="../figures")
    parser.add_argument('--checkpoint_dir', type=str, help="path to checkpoint directory", default="../checkpoints")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing checkpoint")
    parser.add_argument('--rep', type=int, help="index of the repetition", default=0)
    parser.add_argument("--n", type=int, help="number of treatment not assigned", default=1000)
    parser.add_argument("--p", type=int, help="number of features", default=100)
    parser.add_argument("--amplitude", type=int, help="amplitude of weights", default=10)
    parser.add_argument("--non_compliance_type", type=str, help="type of non-compliance", choices=["one-sided", "two-sided"], required=True)
    parser.add_argument("--non_compliance_rate", type=float, help="rate of non-compliance", default=0.5)
    parser.add_argument("--u_ay", type=int, help="number unobserved confounders between A anbd Y", default=0)
    parser.add_argument("--u_ty", type=int, help="number unobserved confounders between T anbd Y", default=0)
    parser.add_argument("--model", type=str, help="choice of base estimator model",
                        choices=["sbd-tlearner", "sbd-dragon", "cfd-tlearner", "lobster"])
    args = parser.parse_args()
    main(args)
