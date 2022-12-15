
import numpy as np
import sys
sys.path.append("..")
from config import *
import time
import os
from tqdm import tqdm
import nibabel as nib
import scipy.ndimage as snd
from widgets import *
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from utils import get_folds_structure, compute_auc_and_threshold, get_dataframe_from_metadata
from scipy.interpolate import interp1d
from statsmodels.stats.contingency_tables import mcnemar


def yielder(results_per_fold):
    for (model_name, evaluation, ensembled), gr in results_per_fold.groupby(["model_name", "evaluation", "ensembled"]):
        
        yield (model_name, evaluation, ensembled, gr["auc"].mean(), gr["tn"].sum(), gr["fp"].sum(), gr["fn"].sum(), gr["tp"].sum(), 
               gr["acc"].mean(), gr["f1"].mean(), gr["tpr"].mean(), gr["fpr"].mean(), gr["ppv"].mean(), gr["npv"].mean())
    

def load_results():
    available_models = []
    for root, subdirs, files in os.walk(MODELS_LOAD_FROM):
        for file in files:
            if file.split(".")[1] != "csv":
                continue
            splits = file.split(".")[0].split("-")
            assert len(splits) >= 2 and len(splits) < 5
            available_models.append((splits[0], splits[1], len(splits) > 2 and splits[2] == "ENS", os.path.join(root, file)))
    available_models = pd.DataFrame(available_models, columns = ["model_name", "evaluation", "ensembled", "filename"]).sort_values(["evaluation", "model_name"])

    if not os.path.exists(MODELS_FIGS_SAVE_TO):
        os.mkdir(MODELS_FIGS_SAVE_TO)

    print(f"{len(available_models.index)} models available.")
    available_models.head()
    
    dfs = []
    for i in range(len(available_models.index)):
        df = pd.read_csv(available_models.iloc[i]["filename"])
        df["dataset"] = df["database"]
        df = df[["fold", "dataset", "patient", "lesion", "unseen", "truth", "pred"]]
        df.insert(0, "ensembled", available_models.iloc[i]["ensembled"])
        df.insert(0, "evaluation", available_models.iloc[i]["evaluation"])
        df.insert(0, "model_name", available_models.iloc[i]["model_name"])
        dfs.append(df)
    lw_results = pd.concat(dfs)
    
    return lw_results

def compute_metrics(truth, preds):
    assert set([0,1] + list(np.unique(truth))) == {0,1} and set([0,1] + list(np.unique(preds))) == {0,1}
    #assert list(np.unique(truth)) == [0,1] and list(np.unique(preds))
    tn, fp, fn, tp = metrics.confusion_matrix(truth, preds, labels=(0,1)).ravel()
    fpr = fp / (tn + fp) # 1 - specificity
    tpr = tp / (fn + tp) # sensitivity
    ppv = tp / (tp + fp) # positive predictive value
    npv = tn / (tn + fn) # negative predictive value
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (ppv * tpr) / (ppv + tpr)
    return (tn, fp, fn, tp, acc, f1, tpr, fpr, ppv, npv)
    # columns = ("tn", "fp", "fn", "tp", "acc", "f1", "tpr", "fpr", "ppv", "npv")
    


def compute_lw_model_results(lw_results, accepted_fpr):
    pw_results = []
    lw_th_results_per_fold = []
    lw_th_results = lw_results.copy()
    lw_th_results["pred"] = 0
    for (model_name, evaluation, ensembled), grouped in lw_results[lw_results["unseen"] == 1].groupby(["model_name", "evaluation", "ensembled"]):
        for fold, grouped_by_fold in grouped.groupby("fold"):
            
            auc, th = compute_auc_and_threshold(grouped_by_fold["truth"], grouped_by_fold["pred"], accepted_fpr)
            grouped_by_fold.insert(len(grouped_by_fold.columns)-1, "pred_th", 0)
            grouped_by_fold["pred_th"] = 0
            grouped_by_fold.loc[grouped_by_fold["pred"] >= th, "pred_th"] = 1
            
            lw_th_results.loc[(lw_results["unseen"] == 1) &
                           (lw_results["model_name"] == model_name) &
                           (lw_results["evaluation"] == evaluation) &
                           (lw_results["ensembled"] == ensembled) &
                           (lw_results["fold"] == fold) &
                           (lw_results["pred"] >= th), "pred"] = 1
            
            lw_th_results_per_fold.append((model_name, evaluation, ensembled, fold, th, auc) + compute_metrics(grouped_by_fold["truth"], grouped_by_fold["pred_th"]))


    # lw results per fold/model
    lw_th_results_per_fold = pd.DataFrame(data = lw_th_results_per_fold, 
                                 columns = ("model_name", "evaluation", "ensembled", "fold", "th", "auc", "tn", "fp", "fn", "tp", "acc", "f1", "tpr", "fpr", "ppv", "npv"))
    lw_th_results_per_fold[lw_th_results_per_fold["evaluation"] == "basel"].sort_values("acc")
    
    # lw results per model
    lw_th_results_per_model = pd.DataFrame(data = yielder(lw_th_results_per_fold), 
             columns = ("model_name", "evaluation", "ensembled", "auc", "tn", "fp", "fn", "tp", "acc", "f1", "tpr", "fpr", "ppv", "npv"))

    
    # add EXPERTS
    for fv in lw_th_results_per_model["evaluation"].unique():
        votes = get_votes_lw(lw_th_results, fv)
        lw_th_per_expert = pd.DataFrame(data=[
                            ("Pietro", fv, True) + compute_metrics(votes["truth"], votes["Pietro"]), 
                          ("Martina", fv, True) + compute_metrics(votes["truth"], votes["Martina"]), ], 
                     columns=["model_name", "evaluation", "ensembled", "tn", "fp", "fn", "tp", "acc", "f1", "tpr", "fpr", "ppv", "npv"])

        lw_th_results_per_model = pd.concat([lw_th_results_per_model, lw_th_per_expert], sort=False)
    
    return lw_th_results, lw_th_results_per_fold, lw_th_results_per_model.sort_values(["evaluation", "auc"], ascending=[True, False])

def compute_pw_model_results(lw_results, accepted_fpr, chronicity_th):
    pw_results = []
    pw_results_per_fold = []
    for (model_name, evaluation, ensembled), grouped in lw_results[lw_results["unseen"] == 1].groupby(["model_name", "evaluation", "ensembled"]):

        for fold, grouped_by_fold in grouped.groupby("fold"):
            auc, th = compute_auc_and_threshold(grouped_by_fold["truth"], grouped_by_fold["pred"], accepted_fpr)
            grouped_by_fold.insert(len(grouped_by_fold.columns)-1, "pred_th", 0)
            grouped_by_fold.loc[grouped_by_fold["pred"] >= th, "pred_th"] = 1
            
            # PW
            pw_fold_results = []
            for patient, grouped_by_pat in grouped_by_fold.groupby(["patient"]):
                pw_fold_results.append([model_name, evaluation, ensembled, fold, patient, grouped_by_pat["truth"].sum(), grouped_by_pat["pred_th"].sum()])
            pw_fold_results = pd.DataFrame(data = pw_fold_results, 
                                 columns = ("model_name", "evaluation", "ensembled", "fold", "patient", "truth_rims", "pred_rims"))
            pw_results.append(pw_fold_results)

            pw_fold_results["truth"] = (pw_fold_results["truth_rims"] >= chronicity_th).astype(int)
            pw_fold_results["pred"] = (pw_fold_results["pred_rims"] >= chronicity_th).astype(int)
            pw_results_per_fold.append((model_name, evaluation, ensembled, fold) + compute_metrics(pw_fold_results["truth"], pw_fold_results["pred"]))

    pw_results_per_fold = pd.DataFrame(data = pw_results_per_fold, 
                                 columns = ("model_name", "evaluation", "ensembled", "fold", "tn", "fp", "fn", "tp", "acc", "f1", "tpr", "fpr", "ppv", "npv"))
    pw_results_per_fold[pw_results_per_fold["evaluation"] == "basel"].sort_values("acc")


    pw_results = pd.concat(pw_results).sort_values(["model_name", "patient"])
    
    pw_model_results = pd.DataFrame(data = [(model_name, evaluation, ensembled) + 
                     compute_metrics(grouped["truth"], grouped["pred"]) 
                     for (model_name, evaluation, ensembled), grouped 
                     in pw_results.groupby(["model_name", "evaluation", "ensembled"])],
            columns = ("model_name", "evaluation", "ensembled", "tn", "fp", "fn", "tp", "acc", "f1", "tpr", "fpr", "ppv", "npv"))
    
     # add EXPERTS
    for fv in pw_model_results["evaluation"].unique():
        votes = get_votes_pw(lw_results, fv, chronicity_th)
        pw_per_expert = pd.DataFrame(data=[
                            ("Pietro", fv, True) + compute_metrics(votes["truth"], votes["Pietro"]), 
                          ("Martina", fv, True) + compute_metrics(votes["truth"], votes["Martina"]), ], 
                     columns=["model_name", "evaluation", "ensembled", "tn", "fp", "fn", "tp", "acc", "f1", "tpr", "fpr", "ppv", "npv"])

        pw_model_results = pd.concat([pw_model_results, pw_per_expert], sort=False)
    
    pw_model_results = pw_model_results.sort_values(["evaluation", "acc"], ascending=[True, False])
    
    return pw_results, pw_results_per_fold, pw_model_results


def get_votes_lw(lw_results, fv):
    fold_patients = pd.DataFrame(data = [(int(db), int(pat)) for l in get_folds_structure(fv) for (db, pat) in l], columns=["dataset", "patient"]).drop_duplicates().sort_values(["dataset", "patient"])
    lesions_list = lw_results[["dataset", "patient", "lesion"]].drop_duplicates()
    lesions_list = pd.merge(fold_patients, lesions_list, on=['dataset','patient'], how='left')
    
    votes_df = pd.read_excel(os.path.join(ROOT_DIR,'votes_preconsensus.xlsx'), sheet_name='Lesion results')[["Patient", "Lesion_ID", "Martina", "Pietro"]]
    votes_df["lesion"] = votes_df["Lesion_ID"]
    votes_df["patient"] = votes_df["Patient"]
    votes_df = votes_df[["patient", "lesion", "Martina", "Pietro"]]
    patients = list(votes_df["patient"])

    # we fix the csv
    counter = 0
    last = 0
    for i in range(len(patients)):
        if np.isnan(patients[i]):
            patients[i] = last
        else:
            last = patients[i]
    votes_df["patient"] = np.array(patients, dtype=int)
    votes_df = votes_df.fillna(0)
    
    lesions_list = pd.merge(lesions_list, votes_df, on=['patient','lesion'], how='left')
    lesions_list = lesions_list.fillna(0)
    lesions_list["truth"] = lesions_list["lesion"] // 1000 == 1
    return lesions_list

def get_votes_pw(lw_results, fv, CHRONIC_TH):
    fold_patients = pd.DataFrame(data = [(int(db), int(pat)) for l in get_folds_structure(fv) for (db, pat) in l], columns=["dataset", "patient"]).drop_duplicates().sort_values(["dataset", "patient"])
    lesions_list = lw_results[["dataset", "patient"]].drop_duplicates()
    lesions_list = pd.merge(fold_patients, lesions_list, on=['dataset','patient'], how='left')
    
    votes_lw = get_votes_lw(lw_results, fv)
    df = pd.DataFrame(data = [(pat, int(grouped["Martina"].sum() >= CHRONIC_TH), 
                               int(grouped["Pietro"].sum() >= CHRONIC_TH), 
                               int((grouped["lesion"] // 1000 == 1).sum() >= CHRONIC_TH)
                              ) 
            for pat, grouped in votes_lw.groupby("patient")], columns = ["patient", "Martina", "Pietro", "truth"])
    
    lesions_list = pd.merge(lesions_list, df, on=['patient',], how='left')
    lesions_list = lesions_list.fillna(0)
    
    return lesions_list


def compare_lw_models(lw_results, model_0, model_1):
    assert model_0[1] == model_1[1] # same eval
    lw_results = lw_results.loc[lw_results["unseen"] == 1]
    #m0 = lw_results.loc[(lw_results["model_name"] == model_0[0]) & (lw_results["evaluation"] == model_0[1]) & (lw_results["ensembled"] == model_0[2]), "pred"]
    
    if model_0[0] == "Pietro" or model_0[0] == "Martina":
        votes = get_votes_lw(lw_results, model_0[1])
        m0 = votes[model_0[0]]
    else:
        m0 = lw_results.loc[(lw_results["model_name"] == model_0[0]) & (lw_results["evaluation"] == model_0[1]) & (lw_results["ensembled"] == model_0[2]), "pred"]
    
    if model_1[0] == "Pietro" or model_1[0] == "Martina":
        votes = get_votes_lw(lw_results, model_0[1])
        m1 = votes[model_1[0]]
    else:
        m1 = lw_results.loc[(lw_results["model_name"] == model_1[0]) & (lw_results["evaluation"] == model_1[1]) & (lw_results["ensembled"] == model_1[2]), "pred"]

    mcnemar_test(m0, m1)

def compare_pw_models(lw_results, pw_results, model_0, model_1, chronic_th):
    assert model_0[1] == model_1[1] # same eval
    lw_results = lw_results.loc[lw_results["unseen"] == 1]
    #m0 = pw_results.loc[(pw_results["model_name"] == model_0[0]) & (pw_results["evaluation"] == model_0[1]) & (pw_results["ensembled"] == model_0[2]), "pred"]
    
    if model_0[0] == "Pietro" or model_0[0] == "Martina":
        votes = get_votes_pw(lw_results, model_0[1], chronic_th)
        m0 = votes[model_0[0]]
    else:
        m0 = pw_results.loc[(pw_results["model_name"] == model_0[0]) & (pw_results["evaluation"] == model_0[1]) & (pw_results["ensembled"] == model_0[2]), "pred"]
    
    
    if model_1[0] == "Pietro" or model_1[0] == "Martina":
        votes = get_votes_pw(lw_results, model_1[1], chronic_th)
        m1 = votes[model_1[0]]
    else:
        m1 = pw_results.loc[(pw_results["model_name"] == model_1[0]) & (pw_results["evaluation"] == model_1[1]) & (pw_results["ensembled"] == model_1[2]), "pred"]
    
    mcnemar_test(m0, m1)

def mcnemar_test(m0, m1):
    table = metrics.confusion_matrix(m0, m1)
    print(table)

    # calculate mcnemar test
    result = mcnemar(table, exact=True)

    # summarize the finding
    print('statistic=%.5f, p-value=%.5f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')
        
        

def compute_curves(all_results, fig_name, config):
    roc_curves = []
    pr_curves = []
    
    #plt.show()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    for (model_name, evaluation, ensembled, color, linestyle) in config:
        model_results = all_results.loc[(all_results["model_name"] == model_name) & (all_results["evaluation"] == evaluation) & (all_results["ensembled"] == ensembled)]
        if len(model_results.index) == 0:
            raise Exception(f"Model not found: {model_name} - {evaluation} - {ensembled}")
         
        num_folds = len(get_folds_structure(version=evaluation)) if not bool(ensembled) else 1
        total_tn, total_fp, total_fn, total_tp, av_th, av_fpr, av_tpr, av_auc, av_auc_pr  = 0, 0, 0, 0, 0, 0, 0, 0, 0
        f_roc = []
        f_pr = []
        for fold in range(num_folds):
            fold_unseen = model_results.loc[(model_results["fold"] == fold) & (model_results["unseen"])]
            total_truths = fold_unseen["truth"]
            total_preds = fold_unseen["pred"]
            
            fpr, tpr, thresholds = metrics.roc_curve(total_truths, total_preds)
            auc = metrics.auc(fpr, tpr)
            av_auc += auc / num_folds
            f_roc.append(interp1d(fpr, tpr, kind='nearest'))
            precision, recall, thresholds_pr = metrics.precision_recall_curve(total_truths, total_preds)
            av_auc_pr += metrics.auc(recall, precision) / num_folds
            f_pr.append(interp1d(recall, precision, kind='nearest'))
            
        # ROC
        xs = [x/100 for x in range(0, 101)]
        ys = [np.sum([f(x) for f in f_roc])/len(f_roc) for x in xs]
        
        # ROC curves
        ax[0].plot(xs, ys, linestyle, color=color, label=f"{model_name} (AUC={av_auc:.3f})", zorder=1)
        
        # PR
        ys = [np.sum([f(x) for f in f_pr])/len(f_pr) for x in xs]
        ax[1].plot(xs, ys, linestyle, color=color, label=f"{model_name} (AUC={av_auc_pr:.3f})", zorder=1)
    
    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[0].set_xlabel('False positive rate (FPR)')
    ax[0].set_ylabel('True positive rate (TPR)')
    ax[0].set_ylim((-0.05, 1.05))
    ax[0].set_title(f'ROC curve')
    ax[0].legend(loc='best')
        
    # PR curves
    recall_random = [0, 1]
    precision_random = [0.1, 0.1]
    ax[1].plot(recall_random, precision_random, 'k--')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_ylim((-0.05, 1.05))
    ax[1].set_title(f'PR curve')
    ax[1].legend(loc='best')
    
    plt.savefig(os.path.join(MODELS_FIGS_SAVE_TO, f'curves_{fig_name}.pdf'), dpi = 400, format="pdf", bbox_inches='tight')
    


def print_confusion_matrix(r, model, d, fpr, chronicity_th = None):
    if d != "LW" and d != "PW":
        print(f"Mode {d} not supported.")
        return
    elif d == "LW":
        labels = ["Rim-", f"Rim+"]
    else:
        if chronicity_th == None:
            print("Chronicity threshold needed.")
            return
        labels = ["Non-chronic", f"Chronic (>= {chronicity_th})"]
        
    model_name, evaluation, ensembled = model
    r = r[(r["model_name"] == model_name) & 
                      (r["evaluation"] == evaluation) & 
                      (r["ensembled"] == ensembled)][["tn", "fp", "fn", "tp"]]
    data = ((r["tn"].iloc[0], r["fp"].iloc[0]), (r["fn"].iloc[0], r["tp"].iloc[0]))
    
    fig, ax = plt.subplots(1, 1)

    cax = ax.matshow(data, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Predicted", fontsize='x-large')
    ax.set_ylabel("Truth", rotation=0, fontsize='x-large')
    ax.set_xticklabels(['']+labels)
    ax.set_yticklabels(['']+labels)

    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0', pad=0.3), fontsize='x-large')

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_FIGS_SAVE_TO, f'CM{"_PW" + str(chronicity_th) if d == "PW" else "_LW" + str(fpr)}_{model_name}-{evaluation}{"-ENS" if bool(ensembled) else ""}.pdf'), dpi = 400, format="pdf", bbox_inches='tight')
    plt.show()
    plt.clf()