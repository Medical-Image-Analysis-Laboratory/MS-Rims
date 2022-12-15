from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from training import generate_epoch_batches, get_folded_data
from config import *
import numpy as np
from utils import load_lesions, read_patients_metadata, compute_auc_and_threshold
import os
#import tensorflow as tfx

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

BATCH_SIZE = 64

def run_test(configs, from_segmentation=False, split_version=None, test=False):
    if from_segmentation and split_version is None:
        raise Exception("No split version was declared.")
    lpps = [load_lesions(PATCH_SIZE, only_cleaned=False, from_segmentation=from_segmentation, split_version=split_version), ]
    
    models = {}
    for config in configs:
        counter = 0
        
        network_name = config["network_name"]
        models[network_name] = []
        contrasts = config["contrasts"]
        network = config["network"]
        
        print(f"\n[INIT] Loading testing folds for '{network_name}'")
        folds_data = get_folded_data(lpps, 
                                          data_augmentation = "none", 
                                          normalization_type = config["normalization_type"],
                                          contrasts = contrasts,
                                          folds_version = config["folds_version"])

        tf.reset_default_graph()
        g = tf.Graph()
        with g.as_default():
            x, y, lr, eval_dict, cost, optimizer, pred = network.get_model_graph(PATCH_SIZE_TRAINING, n_channels = len(contrasts))
        
        for num_fold in range(len(folds_data)):
            if len(list(folds_data[num_fold]["labels"])) == 0:
                print(f"Fold {num_fold} dismissed because of lack of patients.")
                continue
            images_test = np.concatenate([folds_data[i]["images"] for i in range(len(folds_data)) if len(list(folds_data[i]["labels"])) != 0])
            labels_test = np.concatenate([folds_data[i]["labels"] for i in range(len(folds_data)) if len(list(folds_data[i]["labels"])) != 0])
            meta_test = np.concatenate([folds_data[i]["meta"] for i in range(len(folds_data)) if len(list(folds_data[i]["labels"])) != 0])
            # label for training (false) or testing (true)
            unseen_test = np.array([i == num_fold for i in range(len(folds_data)) if len(list(folds_data[i]["labels"])) != 0 for j in range(folds_data[i]["images"].shape[0])])
            preds = []

            if not test:
                with tf.Session(graph=g) as sess:
                    # Initialization
                    init = tf.initializers.global_variables()
                    sess.run(init)

                    checkpoints_path = os.path.join(PATH_CHECKPOINTS, network_name, str(num_fold))
                    saver = tf.train.Saver(max_to_keep=1)

                    # LOAD CHECKPOINT OF THE FOLD
                    for file in os.listdir(checkpoints_path):
                        if file.endswith(".index"):
                            filename = file.split(".")[0] # with no extension
                            saver.restore(sess, os.path.join(checkpoints_path, filename))

                    # TESTING SET EVALUATION
                    for iteration, (image_batch, label_batch) in enumerate(generate_epoch_batches(images_test, labels_test, BATCH_SIZE, PATCH_SIZE_TRAINING, da_strategy=None,
                                                                                                  use_all=True, random=False)):
                        feed_dict = {x: image_batch, y:label_batch}
                        predictions,  = sess.run([pred, ], feed_dict=feed_dict)
                        preds += [p[1] for p in predictions]
            else:
                preds += [np.random.random() for p in range(labels_test.shape[0])]
                    
            models[network_name].append({
                "preds": preds, "truth": labels_test[:,1], "meta": meta_test, "unseen": unseen_test
            })
                
            counter += len(preds)
            
        to_ensemble = "ensemble" in config and config["ensemble"]
        if to_ensemble:
            preds = np.array(models[network_name][0]["preds"]) / len(models[network_name])
            for i in range(1, len(models[network_name])):
                assert len(models[network_name][0]["preds"]) == len(models[network_name][i]["preds"])
                preds += np.array(models[network_name][i]["preds"]) / len(models[network_name])
            assert (preds <= 1.0).all()
            models[network_name] = [{
                "preds": list(preds), "truth": models[network_name][0]["truth"], "meta": models[network_name][0]["meta"], "unseen": np.array([1 for i in range(models[network_name][0]["truth"].shape[0])])
            }, ]
            
        #if "to_save" in config and config["to_save"]:
        # We save the result in a csv file to evaluate a posteriori
        if not os.path.exists(PATH_TEST_PREDS):
            os.makedirs(PATH_TEST_PREDS)
        # we save a dataframe
        dfs = []
        for i in range(len(models[network_name])):
            f = models[network_name][i]
            m = list(zip(*f["meta"]))
            folds = [i for a in range(f["unseen"].shape[0])]
            dfs.append(pd.DataFrame(data=zip(folds, m[0], m[1], m[2], f["unseen"], f["truth"], f["preds"]), columns = ("fold", "database", "patient", "lesion", "unseen", "truth", "pred")))
        df = pd.concat(dfs).sort_values(by=['database', "patient", "lesion", "fold"])
        fv = config["folds_version"]
        csv_name = f'{network_name}-{fv}'
        if from_segmentation:
            csv_name += f'_{split_version}'
        if to_ensemble:
            csv_name += '-ENS'
        df.to_csv(os.path.join(PATH_TEST_PREDS, csv_name + ".csv"), index=False)
            
        print(f"[{network_name}] Evaluated {counter} lesions.")
    #return models
    

def compute_pw_metrics_segm(results, chronic_thresholds, to_test_th):
    metadata = read_patients_metadata(PATCH_SIZE)
    patients_truth = {}
    for i in metadata:
        for pat in metadata[i]:
            unique_id = f"9{int(i)}{int(pat):04d}"
            patients_truth[unique_id] = 0
            for les in metadata[i][pat]:
                if int(les) // 1000 == 1:# and not bool(metadata[i][pat][les]["ignore"]):
                    patients_truth[unique_id] += 1
    
    for model_name in results:
        if model_name not in to_test_th.keys():
            continue
        results_m = results[model_name]
        patients = {}
        for fold in range(len(results_m)):
            fold_results = results_m[fold]
            total_preds = fold_results["preds"]
            th = to_test_th[model_name]
            #print(th)
            for i in range(len(fold_results["meta"])):
                db, pat, les = fold_results["meta"][i]
                pred = fold_results["preds"][i]
                unique_id = f"9{db}{pat:04d}"
                
                if unique_id not in patients:
                    patients[unique_id] = 0
                if pred >= th:
                    patients[unique_id] += 1
                    
        cols = 2
        rows = len(chronic_thresholds) // cols
        fig, axis = plt.subplots(rows, cols, figsize=(rows * 4, cols * 4))
        for i in range(len(chronic_thresholds)):
            ax = axis[i // 2, i % 2]
            CHRONIC_TH = chronic_thresholds[i]
            assert len(patients_truth.keys()) == len(patients.keys())
            truth = [patients_truth[p] >= CHRONIC_TH for p in patients_truth]
            preds = [patients[p] >= CHRONIC_TH for p in patients]
        
            data = metrics.confusion_matrix(truth, preds)
            labels = ["Non-chronic", f"Chronic (>= {CHRONIC_TH})"]

            cax = ax.matshow(data, cmap=plt.cm.Blues)
            #fig.colorbar(cax, cax=ax)

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
        #plt.show()
        plt.savefig(os.path.join(PATH_TEST, f'PW_{model_name}_SEG.pdf'), dpi = 400, format="pdf", bbox_inches='tight')
        
        plt.clf()
