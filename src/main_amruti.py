import os, argparse, random, scipy
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import plot_binary_estimation, plot_auroc, rmse, plot_binary_effect_estimates
from data.amr_uti import generate_amr_uti_nc
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
      data_dir = os.path.join(args.data_dir, "AMR-UTI")
      assert os.path.isdir(data_dir)
      model_str = f"{args.model}"
      if args.non_compliance_type == "one-sided": assert len(args.prescriptions) == 1
      elif args.non_compliance_type == "two-sided": assert len(args.prescriptions) == 2
      data_str = f"type={args.non_compliance_type}/rate={args.non_compliance_rate}/prescriptions={args.prescriptions}"
      data_rpath = os.path.join("AMR-UTI", data_str, f"rep={args.rep}")
      checkpoint_dir = os.path.join(args.checkpoint_dir, data_rpath, model_str)
      os.makedirs(checkpoint_dir, exist_ok=True)
      model_checkpoint_dir = os.path.join(checkpoint_dir, "model")
      os.makedirs(model_checkpoint_dir, exist_ok=True)
      figure_dir = os.path.join(args.figure_dir, data_rpath, model_str)
      os.makedirs(figure_dir, exist_ok=True)
      
      ### Load & preprocess data ###
      train_df, test_df, X_cols = generate_amr_uti_nc(
            data_dir, rep=args.rep, prescriptions=args.prescriptions, 
            nc_type=args.non_compliance_type, nc_rate=args.non_compliance_rate)
      delta_A = np.abs(np.mean(test_df["A_T1"] - test_df["A_T0"]))
      
      print(f"#train={len(train_df)}, #test={len(test_df)}")
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
      batch_size_list=[4096*2]
      reg_l2_list=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 0.0]
      learning_rate_list=[1e-4]
      early_stop=True
      epochs=300
      is_sbd = "sbd" in args.model
      if is_sbd:
            if args.model == "sbd-tlearner":
                  model, metric_dict, hparam_dict = train_tlearner_grid_search(X_train, T_train, Y_train, checkpoint_dir=checkpoint_dir, 
                  overwrite=args.overwrite, batch_size_list=batch_size_list, verbose=False, early_stop=early_stop, 
                  reg_l2_list=reg_l2_list, learning_rate_list=learning_rate_list, epochs=epochs, po_act="sigmoid")
                  Yt_pred_test = tlearner_inference(model, X_test, batch_size=hparam_dict["batch_size"])
            elif args.model == "sbd-dragon":
                  model, metric_dict, hparam_dict = train_dragon_grid_search(X_train, T_train, Y_train, checkpoint_dir=checkpoint_dir, 
                  overwrite=args.overwrite, batch_size_list=batch_size_list, verbose=True, epochs=epochs, po_act="sigmoid",
                  early_stop=early_stop, reg_l2_list=reg_l2_list, learning_rate_list=learning_rate_list)
                  Yt_pred_test, _ = dragon_inference(model, X_test, batch_size=hparam_dict["batch_size"])
      else:
            if args.model == "cfd-tlearner":
                  model_t, model_a, model_y_t0, model_y_t1, metric_dict, hparam_dict = train_cfd_tlearner_grid_search(
                        X_train, A_train, T_train, Y_train, early_stop=early_stop, verbose=True, 
                        po_act="sigmoid", checkpoint_dir=model_checkpoint_dir, overwrite=args.overwrite, 
                        batch_size_list=batch_size_list, reg_l2_list=reg_l2_list, learning_rate_list=learning_rate_list, 
                        epochs=epochs, shared_backbone=False)
                  t_pred_test = mlp_inference(model_t, X_test, batch_size=hparam_dict["best_t_model_hparam"]["batch_size"])
                  a_pred_test = tlearner_inference(model_a, X_test, batch_size=hparam_dict["best_a_model_hparam"]["batch_size"])
                  y_pred_t0_test = tlearner_inference(model_y_t0, X_test, batch_size=hparam_dict["best_y_t0_model_hparam"]["batch_size"])
                  y_pred_t1_test = tlearner_inference(model_y_t1, X_test, batch_size=hparam_dict["best_y_t1_model_hparam"]["batch_size"])
                  y_pred_test = np.hstack((y_pred_t0_test, y_pred_t1_test))
            elif args.model == "lobster":
                  model, metric_dict, hparam_dict = train_lobster_grid_search(X_train, A_train, T_train, Y_train, batch_size_list=batch_size_list, 
                        reg_l2_list=reg_l2_list, checkpoint_dir=model_checkpoint_dir, overwrite=args.overwrite, early_stop=early_stop, 
                        verbose=True, po_act="sigmoid", learning_rate_list=learning_rate_list, epochs=epochs)
                  t_pred_test, a_pred_test, y_pred_test = lobster_inference(model, X_test, batch_size=hparam_dict["batch_size"])
            Ya_pred_test, Yt_pred_test = cfd_inference(t_pred_test, a_pred_test, y_pred_test, non_compliance_type=args.non_compliance_type)

      # plot loss
      fig, axes = plt.subplots(1, 2, figsize=(10, 5))
      if args.model == "sbd-dragon":
            axes[0].plot(metric_dict["val_loss"], label="overall loss")
            axes[0].plot(metric_dict["val_outcome_loss"], label="outcome loss")
            axes[0].plot(metric_dict["val_treatment_loss"], label="treatment loss")
            axes[1].plot(metric_dict["val_treatment_accuracy"], label="treatment accuracy")
            axes[1].plot(metric_dict["val_outcome_accuracy"], label="outcome accuracy")
      elif args.model.startswith("sbd-"):
            axes[0].plot(metric_dict["val_loss"], label="overall loss")
            axes[1].plot(metric_dict["val_outcome_accuracy"], label="outcome accuracy")
      else:
            if args.model.startswith("lobster"):
                  axes[0].plot(metric_dict["val_outcome_loss"], label="outcome loss")
                  axes[1].plot(metric_dict["val_outcome_accuracy"], label="outcome accuracy")
                  axes[0].plot(metric_dict["val_loss"], label="total loss")
            else:
                  axes[0].plot(metric_dict["val_t0_outcome_loss"], label="t0 outcome loss")
                  axes[0].plot(metric_dict["val_t1_outcome_loss"], label="t1 outcome loss")
                  axes[1].plot(metric_dict["val_t0_outcome_accuracy"], label="t0 outcome accuracy")
                  axes[1].plot(metric_dict["val_t1_outcome_accuracy"], label="t1 outcome accuracy")
            axes[0].plot(metric_dict["val_compliance_loss"], label="compliance loss")
            axes[0].plot(metric_dict["val_treatment_loss"], label="treatment loss")
            axes[0].set_yscale("log")
            axes[0].legend()
            axes[1].plot(metric_dict["val_compliance_accuracy"], label="compliance accuracy")
            axes[1].plot(metric_dict["val_treatment_accuracy"], label="treatment accuracy")
      axes[1].legend()
      fig.tight_layout()
      fig.savefig(os.path.join(figure_dir, "loss.png"))

      # evaluate treatment estimation
      print(f"====== Evaluating Estimation ({model_str}) ======")
      eval_dict = {"value": [], "label": [], "delta_A": []}

      if not is_sbd:
            # treatment intake
            if args.non_compliance_type == "two-sided":
                  a_t0_auroc = roc_auc_score(At_test[:, 0], a_pred_test[:, 0])
                  print(f"t0 intake auroc: {a_t0_auroc:.3f}")
                  eval_dict["value"].append(a_t0_auroc)
                  eval_dict["label"].append("t0 intake AUROC")
                  eval_dict["delta_A"].append(delta_A)
            a_t1_auroc = roc_auc_score(At_test[:, 1], a_pred_test[:, 1])
            print(f"t1 intake auroc: {a_t1_auroc:.3f}")
            eval_dict["value"].append(a_t1_auroc)
            eval_dict["label"].append("t1 intake AUROC")
            eval_dict["delta_A"].append(delta_A)
            intake_fig_path = os.path.join(figure_dir, f"prediction_compliance.png")
            if args.non_compliance_type == "one-sided":
                  fig = plot_auroc(a_pred_test[:, 1], At_test[:, 1], title="Compliance prediction")
            else:      
                  fig = plot_binary_estimation(a_pred_test, At_test, title="Compliance prediction")
            fig.savefig(intake_fig_path)

      yt_pehe = rmse((Yt_pred_test[:, 1]-Yt_pred_test[:, 0]), (Yt_test[:, 1]-Yt_test[:, 0]))
      fig = plot_binary_effect_estimates((Yt_pred_test[:, 1]-Yt_pred_test[:, 0]), (Yt_test[:, 1]-Yt_test[:, 0]))
      fig.savefig(os.path.join(figure_dir, f"assign_effect.png"))
      yt_rpehe = yt_pehe / np.sqrt(np.square(Yt_test[:, 1]-Yt_test[:, 0]).mean())
      eval_dict["value"].append(yt_pehe)
      eval_dict["label"].append("assign effect PEHE")
      eval_dict["delta_A"].append(delta_A)
      eval_dict["value"].append(yt_rpehe)
      eval_dict["label"].append("assign effect relative PEHE")
      eval_dict["delta_A"].append(delta_A)
      print(f"Treatment assign pehe: {yt_pehe:.3f}, r-pehe: {yt_rpehe:.3f}")

      eval_dict = pd.DataFrame(eval_dict)
      eval_dict.to_csv(os.path.join(checkpoint_dir, "eval.csv"), index=False)


if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument('--data_dir', type=str, help="path to data directory", default="../data")
      parser.add_argument('--figure_dir', type=str, help="path to data directory", default="../figures")
      parser.add_argument('--checkpoint_dir', type=str, help="path to checkpoint directory", default="../checkpoints")
      parser.add_argument("--overwrite", action="store_true", help="overwrite existing checkpoint")
      parser.add_argument('--rep', type=int, help="index of the repetition", default=0)
      parser.add_argument('--prescriptions', nargs='+', help='choice of prescription as treatment', choices=['CIP', 'NIT', 'SXT', 'LVX'], required=True)
      parser.add_argument("--non_compliance_type", type=str, help="type of non-compliance", choices=["one-sided", "two-sided"], required=True)
      parser.add_argument("--non_compliance_rate", type=float, help="rate of non-compliance", default=0.5)
      parser.add_argument("--model", type=str, help="choice of base estimator model for CFD", 
                          choices=["sbd-tlearner", "sbd-dragon", "cfd-tlearner", "lobster"])
      args = parser.parse_args()
      main(args)