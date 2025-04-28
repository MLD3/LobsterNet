import os, argparse, random, pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data.ihdp import generate_ihdp_nc, generate_ihdp_share
from utils import rmse, plot_continuous_estimation
from estimators.standard import mlp_inference, tlearner_inference, train_tlearner_grid_search
from estimators.propensity import dragon_inference, train_dragon_grid_search
from estimators.frontdoor import cfd_inference, train_lobster_grid_search, lobster_inference, train_cfd_tlearner_grid_search

def main(args):
      ### Initialization ###
      print(f"====== Initialization ======")
      random.seed(args.rep)
      np.random.seed(args.rep)
      tf.random.set_seed(args.rep)
      assert os.path.isdir(args.data_dir)
      rdata_path = os.path.join(args.data_dir, "IHDP", "ihdp.RData")
      assert os.path.isfile(rdata_path)
      if args.type == "rate": data_str = f"rate={args.non_compliance_rate}_size={args.average_effect_size}"
      elif args.type == "share": data_str = f"share={args.sharing_amount}_size={args.average_effect_size}"
      data_rpath = os.path.join(f"IHDP-{args.type}", f"type={args.non_compliance_type}", data_str, f"rep={args.rep}")
      model_str = f"{args.model}"
      checkpoint_dir = os.path.join(args.checkpoint_dir, data_rpath, model_str)
      os.makedirs(checkpoint_dir, exist_ok=True)
      figure_dir = os.path.join(args.figure_dir, data_rpath, model_str)
      os.makedirs(figure_dir, exist_ok=True)
      
      ### Load & preprocess data ###
      if args.type == "rate":
            train_df, test_df = generate_ihdp_nc(rdata_path, rep=args.rep, size=args.average_effect_size,
                  non_compliance_type=args.non_compliance_type, rate=args.non_compliance_rate)
      elif args.type == "share":
            train_df, test_df = generate_ihdp_share(rdata_path, rep=args.rep, size=args.average_effect_size,
                  nc_type=args.non_compliance_type, rho_shared=args.sharing_amount, rho_separate=1-args.sharing_amount)
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
      batch_size_list=[128]
      reg_l2_list=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 0.0]
      learning_rate_list=[1e-4]
      early_stop=True
      epochs=1000
      po_act=None
      is_sbd = "sbd" in args.model
      if is_sbd:
            if args.model == "sbd-tlearner":
                  model, metric_dict, hparam_dict = train_tlearner_grid_search(X_train, T_train, Y_train, checkpoint_dir=checkpoint_dir, 
                  overwrite=args.overwrite, batch_size_list=batch_size_list, verbose=False, early_stop=early_stop, 
                  reg_l2_list=reg_l2_list, learning_rate_list=learning_rate_list, epochs=epochs, po_act=po_act)
                  Yt_pred_test = tlearner_inference(model, X_test, batch_size=hparam_dict["batch_size"])
            if args.model == "sbd-dragon":
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
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            if args.model.startswith("lobster"):
                  axes[0].plot(metric_dict["val_outcome_loss"], label="outcome loss")
                  axes[0].plot(metric_dict["val_loss"], label="overall loss")
            else:
                  axes[0].plot(metric_dict["val_t0_outcome_loss"], label="t0 outcome loss")
                  axes[0].plot(metric_dict["val_t1_outcome_loss"], label="t1 outcome loss")
            axes[0].plot(metric_dict["val_compliance_loss"], label="compliance loss")
            axes[0].plot(metric_dict["val_treatment_loss"], label="treatment loss")
            axes[0].set_yscale("log")
            axes[0].legend()
            axes[1].plot(metric_dict["val_compliance_accuracy"], label="compliance accuracy")
            axes[1].plot(metric_dict["val_treatment_accuracy"], label="treatment accuracy")
            axes[1].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(figure_dir, "loss.png"))
            
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

      print(f"====== Evaluating Estimation ({model_str}) ======")
      eval_dict = {"value": [], "label": []}
      ### evaluate estimations ###
      y_tf_rmse = rmse(Yt_pred_test[range(len(T_test)), T_test], Yt_test[range(len(T_test)), T_test])
      y_tf_rrmse = y_tf_rmse / np.sqrt(np.square(Yt_test[range(len(T_test)), T_test]).mean())
      y_tcf_rmse = rmse(Yt_pred_test[range(len(T_test)), 1-T_test], Yt_test[range(len(T_test)), 1-T_test])
      y_tcf_rrmse = y_tcf_rmse / np.sqrt(np.square(Yt_test[range(len(T_test)), 1-T_test]).mean())
      y_t0_rmse = rmse(Yt_pred_test[:, 0], Yt_test[:, 0])
      y_t0_rrmse = y_t0_rmse / np.sqrt(np.square(Yt_test[:, 0]).mean())
      y_t1_rmse = rmse(Yt_pred_test[:, 1], Yt_test[:, 1])
      y_t1_rrmse = y_t1_rmse / np.sqrt(np.square(Yt_test[:, 1]).mean())
      yt_pehe = rmse((Yt_pred_test[:, 1]-Yt_pred_test[:, 0]), (Yt_test[:, 1]-Yt_test[:, 0]))
      yt_rpehe = yt_pehe / np.sqrt(np.square(Yt_test[:, 1]-Yt_test[:, 0]).mean())
      print(f"Treatment assign factual rmse: {y_tf_rmse:.3f}, counterfactual rmse: {y_tcf_rmse:.3f}, pehe: {yt_pehe:.3f}, r-pehe: {yt_rpehe:.3f}")
      full_stage_fig_path = os.path.join(figure_dir, f"potential_outcome_estimates.png")
      fig = plot_continuous_estimation(Yt_pred_test, Yt_test, title="Potential treatment assignment estimation")
      fig.savefig(full_stage_fig_path)
      eval_dict["value"].append(y_tf_rmse)
      eval_dict["label"].append("factual assign outcome RMSE")
      eval_dict["value"].append(y_tcf_rmse)
      eval_dict["label"].append("counterfactual assign outcome RMSE")
      eval_dict["value"].append(y_tf_rrmse)
      eval_dict["label"].append("factual assign outcome relative RMSE")
      eval_dict["value"].append(y_tcf_rrmse)
      eval_dict["label"].append("counterfactual assign outcome relatrive RMSE")
      eval_dict["value"].append(y_t0_rmse)
      eval_dict["label"].append("treatment not assigned outcome RMSE")
      eval_dict["value"].append(y_t1_rmse)
      eval_dict["label"].append("treatment assigned outcome RMSE")
      eval_dict["value"].append(y_t0_rrmse)
      eval_dict["label"].append("treatment not assigned outcome relative RMSE")
      eval_dict["value"].append(y_t1_rrmse)
      eval_dict["label"].append("treatment assigned outcome relative RMSE")
      eval_dict["value"].append(yt_pehe)
      eval_dict["label"].append("assign effect PEHE")
      eval_dict["value"].append(yt_rpehe)
      eval_dict["label"].append("assign effect relative PEHE")

      eval_dict = pd.DataFrame(eval_dict)
      eval_dict.to_csv(os.path.join(checkpoint_dir, "eval.csv"), index=False)


if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument('--data_dir', type=str, help="path to data directory", default="../data")
      parser.add_argument('--figure_dir', type=str, help="path to data directory", default="../figures")
      parser.add_argument('--checkpoint_dir', type=str, help="path to checkpoint directory", default="../checkpoints")
      parser.add_argument("--overwrite", action="store_true", help="overwrite existing checkpoint")
      parser.add_argument('--rep', type=int, help="index of the repetition", default=0)
      parser.add_argument("--non_compliance_type", type=str, help="type of non-compliance", choices=["one-sided", "two-sided"], required=True)
      parser.add_argument("--average_effect_size", type=float, help="average effect size", default=4.0)
      parser.add_argument("--type", type=str, help="type of IHDP simulation", choices=["rate", "share"], required=True)
      parser.add_argument("--sharing_amount", type=float, help="amount of sharing", default=0.5)
      parser.add_argument("--non_compliance_rate", type=float, help="rate of non-compliance", default=0.5)
      parser.add_argument("--model", type=str, help="choice of base estimator model for CFD", 
                          choices=["sbd-tlearner", "sbd-dragon", "cfd-tlearner", "lobster"])
      args = parser.parse_args()
      main(args)