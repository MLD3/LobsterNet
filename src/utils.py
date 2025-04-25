import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay, RocCurveDisplay

def rmse(a, b): return np.sqrt(np.square(a-b).mean())
def mare(pred, target): return np.abs((target-pred)/target).mean()
def accuracy(a, b): return (a==b).sum() / len(a)

def plot_auroc(pred, target, title=None):
    assert len(pred) == len(target)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    factual_auroc = roc_auc_score(target, pred)
    fpr_f, tpr_f, t_f = roc_curve(target, pred)
    disp_f = RocCurveDisplay(fpr=fpr_f, tpr=tpr_f, roc_auc=factual_auroc)
    disp_f.plot(ax=ax)
    if title: fig.suptitle(title)
    fig.tight_layout()
    return fig

def plot_rmse(pred, target, title=None):
    assert len(pred) == len(target)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(pred, target, ".")
    ax.set_xlabel("prediction")
    ax.set_ylabel("ground truth")
    if title: fig.suptitle(f"{title}, RMSE={rmse(pred, target):.3f}")
    fig.tight_layout()
    return fig


def plot_binary_estimation(pred, target, t=None, plot_effect=False, title=None):
    assert pred.shape[1] == target.shape[1] == 2
    if plot_effect: fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else: fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # plot factual prediction
    if t is None:
        assert pred.shape[0] == target.shape[0]
        factual_auroc = roc_auc_score(target[:, 0], pred[:, 0])
        fpr_f, tpr_f, t_f = roc_curve(target[:, 0], pred[:, 0])
        disp_f = RocCurveDisplay(fpr=fpr_f, tpr=tpr_f, roc_auc=factual_auroc, estimator_name='t=0 prediction')
        disp_f.plot(ax=axes[0])
        axes[0].set_title("Factual Outcome AUROC")
        # plot counterfactual prediction
        couterfactual_auroc = roc_auc_score(target[:, 1], pred[:, 1])
        fpr_cf, tpr_cf, t_cf = roc_curve(target[:, 1], pred[:, 1],)
        disp_cf = RocCurveDisplay(fpr=fpr_cf, tpr=tpr_cf, roc_auc=couterfactual_auroc, estimator_name='t=1 prediction')
        disp_cf.plot(ax=axes[1])
        axes[1].set_title("Counter Factual Outcome AUROC")
    else:
        assert pred.shape[0] == target.shape[0] == t.shape[0]
        factual_auroc = roc_auc_score(target[range(len(t)), t], pred[range(len(t)), t])
        fpr_f, tpr_f, t_f = roc_curve(target[range(len(t)), t], pred[range(len(t)), t])
        disp_f = RocCurveDisplay(fpr=fpr_f, tpr=tpr_f, roc_auc=factual_auroc, estimator_name='factual prediction')
        disp_f.plot(ax=axes[0])
        axes[0].set_title("Factual Outcome AUROC")
        # plot counterfactual prediction
        couterfactual_auroc = roc_auc_score(target[range(len(t)), (1-t)], pred[range(len(t)), (1-t)])
        fpr_cf, tpr_cf, t_cf = roc_curve(target[range(len(t)), 1-t], pred[range(len(t)), 1-t],)
        disp_cf = RocCurveDisplay(fpr=fpr_cf, tpr=tpr_cf, roc_auc=couterfactual_auroc, estimator_name='counterfactual prediction')
        disp_cf.plot(ax=axes[1])
        axes[1].set_title("Counter Factual Outcome AUROC")
    # plot effect prediction
    if plot_effect:
        axes[2].plot(pred[:, 1]-pred[:, 0], target[:, 1]-target[:, 0], ".")
        axes[2].set_xlabel("prediction")
        axes[2].set_ylabel("ground truth")
        spearman = scipy.stats.spearmanr(a=pred[:, 1]-pred[:, 0], b=target[:, 1]-target[:, 0])[0]
        axes[2].set_title(f"Effect spearman: {spearman:.3f}")

    if title: fig.suptitle(title)
    fig.tight_layout()
    return fig
      

def plot_continuous_estimation(pred, target, t=None, plot_effect=True, title=None):
    assert pred.shape[1] == target.shape[1] == 2
    if plot_effect:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if t is None:
        assert pred.shape[0] == target.shape[0]
        # plot factual prediction
        axes[0].plot(pred[:, 0], target[:, 0], ".")
        factual_rmse = rmse(pred[:, 0], target[:, 0])
        axes[0].set_title(f"t=0 prediction RMSE: {factual_rmse:.3f}")
        min_val = min(pred[:, 0].min(), target[:, 0].min())
        max_val = max(pred[:, 0].max(), target[:, 0].max())
        axes[0].set_xlim(min_val-(np.abs(min_val)*0.5), max_val+(np.abs(max_val)*0.5))
        axes[0].set_ylim(min_val-(np.abs(min_val)*0.5), max_val+(np.abs(max_val)*0.5))
        # plot counterfactual prediction
        axes[1].plot(pred[:, 1], target[:, 1], ".")
        counterfactual_rmse = rmse(pred[:, 1], target[:, 1])
        axes[1].set_title(f"t=1 prediction RMSE: {counterfactual_rmse:.3f}")
        min_val = min(pred[:, 1].min(), target[:, 1].min())
        max_val = max(pred[:, 1].max(), target[:, 1].max())
        axes[1].set_xlim(min_val-(np.abs(min_val)*0.5), max_val+(np.abs(max_val)*0.5))
        axes[1].set_ylim(min_val-(np.abs(min_val)*0.5), max_val+(np.abs(max_val)*0.5))
    else:
        assert pred.shape[0] == target.shape[0] == t.shape[0]
        # plot factual prediction
        axes[0].plot(pred[range(len(t)), t], target[range(len(t)), t], ".")
        factual_rmse = rmse(pred[range(len(t)), t], target[range(len(t)), t])
        axes[0].set_title(f"factual prediction RMSE: {factual_rmse:.3f}")
        min_val = min(pred[range(len(t)), t].min(), target[range(len(t)), t].min())
        max_val = max(pred[range(len(t)), t].max(), target[range(len(t)), t].max())
        axes[0].set_xlim(min_val-(np.abs(min_val)*0.3), max_val+(np.abs(max_val)*0.3))
        axes[0].set_ylim(min_val-(np.abs(min_val)*0.3), max_val+(np.abs(max_val)*0.3))
        # plot counterfactual prediction
        axes[1].plot(pred[range(len(t)), 1-t], target[range(len(t)), 1-t], ".")
        counterfactual_rmse = rmse(pred[range(len(t)), 1-t], target[range(len(t)), 1-t])
        axes[1].set_title(f"counter factual prediction RMSE: {counterfactual_rmse:.3f}")
        min_val = min(pred[range(len(t)), 1-t].min(), target[range(len(t)), 1-t].min())
        max_val = max(pred[range(len(t)), 1-t].max(), target[range(len(t)), 1-t].max())
        axes[1].set_xlim(min_val-(np.abs(min_val)*0.3), max_val+(np.abs(max_val)*0.3))
        axes[1].set_ylim(min_val-(np.abs(min_val)*0.3), max_val+(np.abs(max_val)*0.3))
    if plot_effect:
        # plot effect estimation
        axes[2].plot(pred[:, 1]-pred[:, 0], target[:, 1]-target[:, 0], ".")
        pehe = rmse(pred[:, 1]-pred[:, 0], target[:, 1]-target[:, 0])
        r_pehe = pehe/np.sqrt(np.square(target[:, 1]-target[:, 0]).mean())
        axes[2].set_title(f"Effect pehe: {pehe:.3f}, relative pehe: {r_pehe:.3f}")
        min_val = min((pred[:, 1]-pred[:, 0]).min(), (target[:, 1]-target[:, 0]).min())
        max_val = max((pred[:, 1]-pred[:, 0]).max(), (target[:, 1]-target[:, 0]).max())
        axes[2].set_xlim(min_val-(np.abs(min_val)*0.3), max_val+(np.abs(max_val)*0.3))
        axes[2].set_ylim(min_val-(np.abs(min_val)*0.3), max_val+(np.abs(max_val)*0.3))

    for i in range(len(axes)):
        axes[i].set_xlabel("Prediction")
        axes[i].set_ylabel("Ground truth")
    if title: fig.suptitle(title)
    fig.tight_layout()
    return fig

def plot_effect(pred, target, title=None):
    fig, ax = plt.subplots(1, 1)
    ax.plot(pred, target, ".")
    pehe = rmse(pred, target)
    r_pehe = pehe/np.sqrt(np.square(target).mean())
    min_val = min((pred).min(), (target).min())
    max_val = max((pred).max(), (target).max())
    ax.set_xlim(min_val-(np.abs(min_val)*0.3), max_val+(np.abs(max_val)*0.3))
    ax.set_ylim(min_val-(np.abs(min_val)*0.3), max_val+(np.abs(max_val)*0.3))
    ax.set_title(f"title\nEffect pehe: {pehe:.3f}, relative pehe: {r_pehe:.3f}")
    fig.tight_layout()
    return fig



def plot_continuous_potential_outcome_estimates(y_at_pred, y_at_target):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    title_arr = ["T=0, A=0", "T=0, A=1", "T=1, A=0", "T=1, A=1"]
    for i in range(y_at_pred.shape[1]):
        axes[i].plot(y_at_pred[:, i], y_at_target[:, i], ".")
        min_val = min(y_at_pred[:, i].min(), y_at_target[:, i].min())
        max_val = max(y_at_pred[:, i].max(), y_at_target[:, i].max())
        axes[i].set_xlim(min_val-max(1.0, np.abs(min_val)*0.3), max_val+max(1.0, np.abs(max_val)*0.3))
        axes[i].set_ylim(min_val-max(1.0, np.abs(min_val)*0.3), max_val+max(1.0, np.abs(max_val)*0.3))
        curr_rmse = rmse(y_at_pred[:, i], y_at_target[:, i])
        axes[i].set_title(f"rmse: {curr_rmse:.3f}, {title_arr[i]}")
    fig.tight_layout()
    return fig


def plot_binary_potential_outcome_estimates(y_at_pred, y_at_target):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    title_arr = ["T=0, A=0", "T=0, A=1", "T=1, A=0", "T=1, A=1"]
    for i in range(y_at_pred.shape[1]):
        factual_auroc = roc_auc_score(y_at_target[:, i], y_at_pred[:, i])
        fpr_f, tpr_f, t_f = roc_curve(y_at_target[:, i], y_at_pred[:, i])
        disp_f = RocCurveDisplay(fpr=fpr_f, tpr=tpr_f, roc_auc=factual_auroc)
        disp_f.plot(ax=axes[i])
        curr_auroc = roc_auc_score(y_at_target[:, i], y_at_pred[:, i])
        axes[i].set_title(f"auroc: {curr_auroc:.3f}, {title_arr[i]}")
    fig.tight_layout()
    return fig

def plot_binary_effect_estimates(effect_pred, effect_target):
    fig, ax = plt.subplots(1, 1)
    for effect in np.unique(effect_target):
        ax.hist(effect_pred[effect_target==effect], label=f"True effect={effect}", 
            density=True, alpha=0.5)
    pehe = rmse(effect_pred, effect_target)
    ax.set_title(f"PEHE={pehe:.3f}")
    ax.legend()
    fig.tight_layout()
    return fig