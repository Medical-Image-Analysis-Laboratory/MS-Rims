import time
import numpy as np
from scipy import ndimage
import logging
import os
from config import *
from utils import get_folds_structure, normalize_patch, enhance_flair, enhance_flairstar
import tensorflow as tf
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix
from datetime import datetime


BATCH_SIZE = 32
#LEARNING_RATES = [ 0.0001, 0.000075, 0.00005, 0.000025, 0.00001]
LEARNING_RATES = [ 0.0001, 0.00005, 0.000025, 0.00001]
MAX_EPOCHS_LOSING = 3
N_MAX_EPOCHS = 50
EPOCHS_BETWEEN_CHECKS = 1
MAX_CHECKPOINTS_KEPT = 1
LOG_FILENAME = None
LOGS_FOLDER = os.path.join(ROOT_DIR, "logs")



def print_all_2(message):
    print(message)
    logging.info(message)
    
def print_all(message):
    print(message)
    if LOG_FILENAME is not None:
        log_path = os.path.join(LOGS_FOLDER, LOG_FILENAME)
        with open(log_path, "a+") as logfile:
            logfile.write(message)
            logfile.write("\n")

def process_lesion(lesion, contrasts, normalization):
    '''
    Function that receives a lesion read from raw images and transforms it to deep learning format (last dimension for constrast), normalizing each. The contrasts used are the ones specified in "contrasts".
    '''
    width, height, depth = lesion[list(lesion.keys())[0]].shape
    #image = np.zeros((width, height, depth, len(contrasts)), dtype="float32")
    channels = []
    for c in contrasts:
        if c == "eFLAIR" and c in lesion.keys():
            to_add = enhance_flair(lesion["FLAIR"], normalization, blur=False)
        elif c == "eFLAIR_bl" and c in lesion.keys():
            to_add = enhance_flair(lesion["FLAIR"], normalization, blur=True)
        elif c == "eFLAIRSTAR" and c in lesion.keys():
            to_add = enhance_flair(lesion["FLAIRSTAR"], normalization)
        elif c == "MP2RAGE_T1MAP":
            to_add = lesion[c]
        elif WORKING_MODE_SKULL_STRIPPED and c == "MP2RAGE_UNI_SK" and c not in lesion.keys(): # We use MP2RAGE_SYNTHETIC if MP2RAGE_UNI_SK is not available
            if "MP2RAGE_SYNTHETIC" not in lesion.keys():
                raise(Exception(f"Lesions for contrast {c} and MP2RAGE_SYNTHETIC not available.")) 
            #print("MP2RAGE_UNI_SK not found. Using MP2RAGE_SYNTHETIC instead...")
            to_add = normalize_patch(lesion["MP2RAGE_SYNTHETIC"], normalization)
        elif c in lesion.keys():
            to_add = normalize_patch(lesion[c], normalization)
        else:
            raise(Exception(f"Lesions for contrast {c} not available.")) 
        channels.append(to_add)
    return np.stack(channels, axis=-1)

def get_folded_data(lpps, *, normalization_type="local_max", data_augmentation="all", contrasts=["T2STAR_PHASE"], folds_version = "all"):
    '''
        Function that processes all the lesions (load_data) and divides them into folds, by the fold version specified.
        RETURN:
            - folds_data: array with the original lesions.
                    {
                            "images": ...,
                            "labels": ...,
                            "masks": ...
                    }
            - folds_data_DA: array with the lesions result of offline DA (only used in training).
                    {
                            "images": ...,
                            "labels": ...,
                            "masks": ...
                    }
            - weights: array with the weights for each fold.
                    [(1, 0.95), ..., (1, 0.94)]
    '''
    if not all(c in IMPLEMENTED_CONTRASTS for c in contrasts):
        raise(Exception(f"Contrast not supported: {contrasts}."))
        
    folds = get_folds_structure(version = folds_version)
    print_all(f"[INFO] Folds version: {folds_version}.")
    
    folds_data = {}
    for i in range(len(folds)):
        t0 = time.time()
        patches = [(LABEL_RIM if int(les) // 1000 == 1 else LABEL_NON_RIM, 
                    process_lesion(lpp[db_id][int(pat)][les], contrasts, normalization_type),
                   (int(db_id), int(pat), int(les)))
                               for lpp in lpps 
                               for (db_id, pat) in folds[i]
                               for les in lpp[db_id][int(pat)]]
    
        if data_augmentation == "all":
            # We perform the augmentation on RIM LESIONS
            axes = ((0,1), (1,2), (0, 2))
            degrees = (90., 180., 270.)
            
            patches += [(lab, 
                         ndimage.rotate(les, deg, axes=ax, reshape=False, order=0, mode='nearest'),
                           metadata)
                            for ax in axes
                            for deg in degrees
                            for (lab, les, metadata) in patches
                            if lab[1] == 1] # only if rim
            
            #print(f"[INFO] DA performed on fold {i} ({time.time() - t0} secs)...")
            
        folds_data[i] = {
            "images": np.array(list(zip(*patches))[1]) if len(folds[i]) > 0 else [],
            "labels": np.array(list(zip(*patches))[0]) if len(folds[i]) > 0 else [],
            "meta": np.array(list(zip(*patches))[2]) if len(folds[i]) > 0 else [],
        }
        
        n_rims = np.sum(folds_data[i]["labels"][:,1]) if len(folds[i]) > 0 else 0
        n_non_rims = np.sum(folds_data[i]["labels"][:,0]) if len(folds[i]) > 0 else 0
        print_all(f"[F{i}] {n_rims}/{n_non_rims} (+/-). Execution time: {time.time() - t0} secs.")
        
        #print(folds_data[i]["labels"])
        
    return folds_data

def calculate_weight(folds_labels, num_fold):
    count_rim = np.sum([np.sum((folds_labels[j] == LABEL_RIM).astype(int)) 
                        for j in range(len(folds_labels)) if j != num_fold])
    count_non_rim = np.sum([np.sum((folds_labels[j] == LABEL_NON_RIM).astype(int)) 
                        for j in range(len(folds_labels)) if j != num_fold])

    return [1.0, count_non_rim / count_rim]


# TRAINING _____________________________________

def __train(network, images_tr, labels_tr, images_test, labels_test, da_strategy, num_fold, epochs_list, weight_ce, contrasts, net_name="unknown"):
    current_lr_idx = 0

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        x, y, lr, eval_dict, cost, optimizer, pred = network.get_model_graph(PATCH_SIZE_TRAINING,
                                                                     n_channels = len(contrasts),
                                                                     weight_ce=weight_ce)

    with tf.Session(graph=g) as sess:
        # Initialization
        init = tf.initializers.global_variables()
        sess.run(init)

        checkpoints_path = os.path.join(PATH_CHECKPOINTS, net_name, str(num_fold))
        summary_writer = tf.summary.FileWriter(os.path.join(PATH_SUMMARY, net_name, str(num_fold)))#, sess.graph)
        saver = tf.train.Saver(max_to_keep=MAX_CHECKPOINTS_KEPT)

        # Training
        n_epoch = 1
        while(n_epoch <= epochs_list[-1]):
            epoch_time = time.time()
            loss_sum_train = 0
            preds, truth = [], []
            for iteration, (image_batch, label_batch) in tqdm(enumerate(generate_epoch_batches(images_tr, labels_tr, BATCH_SIZE, PATCH_SIZE_TRAINING, da_strategy=da_strategy))):
                loss, _, predictions = sess.run([cost, optimizer, pred], feed_dict={
                    lr: LEARNING_RATES[current_lr_idx],
                    x: image_batch,
                    y: label_batch
                })
                
                loss_sum_train += loss
                preds += [0 if p[0] > p[1] else 1 for p in predictions]
                truth += [int(label_batch[idx, 1]) for idx in range(label_batch.shape[0])]
            epoch_time = time.time() - epoch_time
            # TRAINING SET EVALUATION (with offline DA...)
            tn, fp, fn, tp = confusion_matrix(truth, preds, labels=[0,1]).ravel()
            acc_training = ((tp + tn) / (tp + fn + fp + tn))
            
            # Saving of checkpoint
            if n_epoch % EPOCHS_BETWEEN_CHECKS == 0:
                saver.save(sess, os.path.join(checkpoints_path, "check"), global_step=n_epoch)
            
            eval_time = time.time()
            # TESTING SET EVALUATION
            loss_sum_test = 0
            preds, truth = [], []
            for iteration, (image_batch, label_batch) in enumerate(generate_epoch_batches(images_test, labels_test, BATCH_SIZE, PATCH_SIZE_TRAINING, use_all=True)):
                feed_dict = {x: image_batch, y: label_batch}
                loss, predictions = sess.run([cost, pred], feed_dict=feed_dict)
                loss_sum_test += loss

                preds += [0 if p[0] > p[1] else 1 for p in predictions]
                truth += [int(label_batch[idx, 1]) for idx in range(label_batch.shape[0])]
            tn, fp, fn, tp = confusion_matrix(truth, preds, labels=[0,1]).ravel()
            tpr_test, fpr_test, tnr_test = (tp / (tp + fn)), (fp / (fp + tn)), (tn / (tn + fp))
            acc_testing = ((tp + tn) / (tp + fn + fp + tn))
            
            eval_time = time.time() - eval_time

            loss_train = loss_sum_train / images_tr.shape[0]
            loss_test = loss_sum_test / images_test.shape[0]
            print_all(f"[TR_F{num_fold}_EPOCH_{n_epoch:02d}] LR = {LEARNING_RATES[current_lr_idx]:0.5f} | TRAIN ({int(epoch_time)}): Loss={loss_train:0.5f}, TrAcc={acc_training:0.5f} || TEST ({int(eval_time)}): Loss={loss_test:0.5f}, Acc={acc_testing:0.5f}, TPR={tpr_test:0.5f}, FPR={fpr_test:0.5f}")

            epochs_summary = tf.Summary(value=[
                tf.Summary.Value(tag="training_loss", simple_value= loss_train),
                tf.Summary.Value(tag="testing_loss", simple_value=loss_test),
                tf.Summary.Value(tag="training_accuracy", simple_value=acc_training),
                tf.Summary.Value(tag="testing_accuracy", simple_value=acc_testing),
                tf.Summary.Value(tag="tpr", simple_value=tpr_test),
                tf.Summary.Value(tag="tnr", simple_value=tnr_test),
                tf.Summary.Value(tag="fpr", simple_value=fpr_test)
            ])
            summary_writer.add_summary(epochs_summary, n_epoch)
            summary_writer.flush()
            
            
            while current_lr_idx < len(epochs_list) and epochs_list[current_lr_idx] == n_epoch:
                current_lr_idx += 1
            n_epoch += 1
            

def __train_inner(network, num_fold, images, labels, images_val, labels_val, da_strategy, weight_ce, contrasts, net_name="unknown"):
    epochs_list = []
                
    current_lr_idx = 0
    to_save = os.path.join(PATH_CHECKPOINTS, net_name, f"aux_{num_fold}")
    
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        x, y, lr, eval_dict, cost, optimizer, pred = network.get_model_graph(PATCH_SIZE_TRAINING,
                                                                     n_channels = len(contrasts),
                                                                     weight_ce=weight_ce)
        
    with tf.Session(graph=g) as sess:
        # Initialization
        init = tf.initializers.global_variables()
        sess.run(init)
        
        saver = tf.train.Saver(max_to_keep=MAX_CHECKPOINTS_KEPT)
        
        # Training
        epochs_losing_counter = 0
        last_good_loss_value = float('inf')
        n_epoch = 1
        while n_epoch <= N_MAX_EPOCHS:
            epoch_time = time.time()
            for iteration, (image_batch, label_batch) in tqdm(enumerate(generate_epoch_batches(images, labels, BATCH_SIZE, PATCH_SIZE_TRAINING, da_strategy=da_strategy))):
                loss, _ = sess.run([cost, optimizer], feed_dict={
                    lr: LEARNING_RATES[current_lr_idx],
                    x: image_batch,
                    y: label_batch
                })
            epoch_time = (time.time() - epoch_time)
                
            # TESTS ON VALIDATION SET
            eval_time = time.time()
            loss_sum_test = 0
            for iteration, (image_batch, label_batch) in enumerate(generate_epoch_batches(images_val, labels_val, BATCH_SIZE, PATCH_SIZE_TRAINING, use_all=True)):
                loss_sum_test += sess.run(cost, feed_dict={x: image_batch, y: label_batch})
            loss_test = loss_sum_test / images_val.shape[0]
            eval_time = time.time() - eval_time

            # DISPLAYS TRAINING AND TEST ACCURACIES / LOSSES
            if loss_test > last_good_loss_value:
                print_all(f"[CV%d_EPOCH_%02d (+1)] LR = %0.5f, Testing Loss=%0.6f (%0.1f, %0.1f)"
                  % (num_fold, n_epoch,  LEARNING_RATES[current_lr_idx], loss_test, epoch_time, eval_time))
                epochs_losing_counter += 1
                if epochs_losing_counter == MAX_EPOCHS_LOSING:
                    saver.restore(sess, os.path.join(to_save, f"check-{n_epoch - epochs_losing_counter}"))
                    if current_lr_idx + 1 < len(LEARNING_RATES):
                        print_all(f"[CV{num_fold}_RESTORE] Restoring: epoch {n_epoch - epochs_losing_counter} (LR decreased to {LEARNING_RATES[current_lr_idx+1]})")
                    n_epoch = n_epoch - epochs_losing_counter
                    epochs_list.append(n_epoch)
                    epochs_losing_counter = 0
                    current_lr_idx += 1
            else:
                last_good_loss_value = loss_test
                epochs_losing_counter = 0
                saver.save(sess, os.path.join(to_save, "check"), global_step=n_epoch)
                print_all(f"[CV%d_EPOCH_%02d] LR = %0.5f, Testing Loss=%0.6f (%0.1f, %0.1f)"
                  % (num_fold, n_epoch,  LEARNING_RATES[current_lr_idx], loss_test, epoch_time, eval_time))
            
            n_epoch += 1
            if current_lr_idx == len(LEARNING_RATES):
                break
                
    while len(epochs_list) < len(LEARNING_RATES):
        # In case num max of epochs reached
        epochs_list.append(n_max_epochs)
    return epochs_list

def __cross_validate(network, num_fold, folds_images, folds_labels, da_strategy, contrasts, net_name="unknown"):
    results = np.zeros(len(LEARNING_RATES))
    num_folds = len(folds_images)
    for validat_idx in range(num_folds):
        weight = calculate_weight(folds_labels, validat_idx)
        
        print_all(f"\n[START_CV{validat_idx}] Crossvalidating {validat_idx+1}/{num_folds}...")
        print_all(f"Training: {[(idx, len(folds_labels[idx])) for idx in range(num_folds) if idx != validat_idx]}")
        print_all(f"Validation: {(validat_idx, len(folds_labels[validat_idx]))}")
        print_all(f"Weight: {weight[1]:.3f}\n")
        
        images = np.concatenate([folds_images[idx] for idx in range(num_folds) if idx != validat_idx])
        labels = np.concatenate([folds_labels[idx] for idx in range(num_folds) if idx != validat_idx])
        
        images_valid = folds_images[validat_idx]
        labels_valid = folds_labels[validat_idx]
        
        time0 = time.time()
        epochs_list = __train_inner(network, num_fold, images, labels, images_valid, labels_valid, da_strategy, weight, contrasts, net_name=net_name)
        #epochs_list = [round(np.random.random() * 5) + i * 5 for i in range(len(LEARNING_RATES))]
        print_all(f"\n[EPOCHS_{validat_idx}] {epochs_list} ({(time.time() - time0) / 60:.2f} min)")
        
        results += np.array(epochs_list)
        
    return [int(round(val)) for val in np.array(results) / num_folds]
        

def train_fold_ncv(network, num_fold, folds_training, folds_validation, contrasts=None, da_strategy=None, net_name=None, epochs=None):
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)
    date = datetime.now().strftime('%m%d%y_%H%M')
    global LOG_FILENAME
    LOG_FILENAME = f"log_{net_name}_{num_fold}_{date}.txt"
    
    print_all("[START] " + datetime.now().strftime('%m/%d/%y %Hh:%Mmin'))
    
    if contrasts is None or da_strategy is None or net_name is None:
        raise(Exception("Parameters of 'train_fold_ncv' not valid"))
    
    # Creation of needed folders
    if not os.path.exists(os.path.join(PATH_CHECKPOINTS, net_name, str(num_fold))):
        os.makedirs(os.path.join(PATH_CHECKPOINTS, net_name, str(num_fold)))
    if not os.path.exists(os.path.join(PATH_CHECKPOINTS, net_name, f"aux_{num_fold}")):
        os.makedirs(os.path.join(PATH_CHECKPOINTS, net_name, f"aux_{num_fold}"))
    if not os.path.exists(os.path.join(PATH_SUMMARY, net_name, str(num_fold))):
        os.makedirs(os.path.join(PATH_SUMMARY, net_name, str(num_fold)))
        

    folds_images_training = [folds_training[i]["images"] 
                    for i in range(len(folds_training)) if i != num_fold]
    folds_labels_training = [folds_training[i]["labels"] 
                    for i in range(len(folds_training)) if i != num_fold]
    if epochs is None:
        # MODEL INNER CROSSVALIDATION
        print_all(f"\n[START_F{num_fold}_CV] Inner CV for F{num_fold}")
        lr_decay_epochs = __cross_validate(network, num_fold, folds_images_training, folds_labels_training, da_strategy, contrasts, net_name=net_name)
        #lr_decay_epochs = (1, 2, 3, 4, 5)

        print_all(f"\n[RESULT_F{num_fold}_CV] {lr_decay_epochs}")
    else:
        assert len(epochs) == len(LEARNING_RATES)
        lr_decay_epochs = epochs
        print_all(f"\n[INFO] Learning rate epochs decay loaded: {lr_decay_epochs}")
    
    
    print_all(f"\n[START_F{num_fold}_TR] Training F{num_fold}")
    
    # MODEL TRAINING
    weight = calculate_weight(folds_labels_training, num_fold)
    images_training = np.concatenate(folds_images_training)
    labels_training = np.concatenate(folds_labels_training)
    images_testing = folds_validation[num_fold]["images"]
    labels_testing = folds_validation[num_fold]["labels"]
    
    start = time.time()
    __train(network, images_training, labels_training, images_testing, labels_testing, da_strategy, num_fold, lr_decay_epochs, weight, contrasts, net_name=net_name)
    print_all(f"[RESULT_F{num_fold}_TR] Epochs: {lr_decay_epochs[-1]} - Training time: {(time.time() - start) / 60} minutes.")

    print_all(f"\n[END] {datetime.now().strftime('%m/%d/%y %Hh:%Mmin')}")
    
    
    
def train_fold_ncv_autosplit(network, folds_training, folds_validation, contrasts=None, da_strategy=None, net_name=None, epochs=None):
    num_fold = 0
    
    assert len(folds_training) == 2 and len(folds_validation) == 2
    
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)
    date = datetime.now().strftime('%m%d%y_%H%M')
    global LOG_FILENAME
    LOG_FILENAME = f"log_{net_name}_{num_fold}_{date}.txt"
    
    print_all("[START] " + datetime.now().strftime('%m/%d/%y %Hh:%Mmin'))
    
    if contrasts is None or da_strategy is None or net_name is None:
        raise(Exception("Parameters of 'train_fold_ncv_autosplit' not valid"))
    
    # Creation of needed folders
    if not os.path.exists(os.path.join(PATH_CHECKPOINTS, net_name, str(num_fold))):
        os.makedirs(os.path.join(PATH_CHECKPOINTS, net_name, str(num_fold)))
    if not os.path.exists(os.path.join(PATH_CHECKPOINTS, net_name, f"aux_{num_fold}")):
        os.makedirs(os.path.join(PATH_CHECKPOINTS, net_name, f"aux_{num_fold}"))
    if not os.path.exists(os.path.join(PATH_SUMMARY, net_name, str(num_fold))):
        os.makedirs(os.path.join(PATH_SUMMARY, net_name, str(num_fold)))
        
    
    print_all(f"\n[START] Training...")
    
    # MODEL TRAINING

    images_training = folds_training[0]["images"] 
    labels_training = folds_training[0]["labels"] 
    images_valid = folds_validation[1]["images"] 
    labels_valid = folds_validation[1]["labels"] 
    
    count_rim = np.sum((labels_training == LABEL_RIM).astype(int))
    count_non_rim = np.sum((labels_training == LABEL_NON_RIM).astype(int))
    weight = [1.0, count_non_rim / count_rim]
    
    print_all(f"Training: {images_training.shape}")
    print_all(f"Validation: {images_valid.shape}")
    print_all(f"Weight: {weight[1]:.3f}\n")
    
    start = time.time()
    epochs_list = __train_inner(network, num_fold, images_training, labels_training, images_valid, labels_valid, da_strategy, weight, contrasts, net_name=net_name)
    print_all(f"[RESULT_F{num_fold}_TR] Epochs: {epochs_list[-1]} - Training time: {(time.time() - start) / 60} minutes.")

    print_all(f"\n[END] {datetime.now().strftime('%m/%d/%y %Hh:%Mmin')}")
    
    
    

# SUPPORT TO TRAINING ____________________


def generate_epoch_batches(images, labels, batch_size, used_patch_size, da_strategy="v4", use_all = False, random=True,  mode_inference=False):
    '''
        Function that extracts from "images" and "labels" a batch of size "batch size".
        Here, all lesions are cropped to "used_patch_size", which is the patch size that uses the network.
    ''' 
    n_samples = images.shape[0]
    n_batches = n_samples // batch_size
    
    # the remaining samples
    if use_all and n_samples % batch_size != 0:
        # extra slice to use all of them
        n_batches += 1
    
    if random:
        sample_ids = np.random.permutation(n_samples)
    else:
        sample_ids = list(range(0, n_samples))
        
    for i in range(n_batches):
        inds = slice(i * batch_size, min((i + 1) * batch_size, len(sample_ids)))
        perm = sample_ids[inds]
        image_batch = [augment_sample(im, used_patch_size, strategy=da_strategy) for im in images[perm]]
        if not mode_inference:
            label_batch = labels[perm]
            yield (image_batch, label_batch)
        else:
            yield image_batch
     
    
def augment_sample(image, used_patch_size, strategy="v4"):
    '''
        Function that randomly returns a modified version of the lesion. This is where Data Augmentation is performed.
        See versions of DA in README.
    '''
    trans = (0, 0, 0)
    if strategy is None or strategy == 'v1':
        augmented_image = image
    
    else: # v2, v3, v4
        augmented_image = image
        
        if strategy == 'v2' or strategy == 'v3':
            augmented_image = ndimage.rotate(image.copy(), 90 * random.randint(0,3), axes=(0,2), reshape=False, order=0, mode='nearest')
            
        if strategy == 'v5' or strategy == 'v6':
            ndimage.rotate(image.copy(), 90 * random.randint(0,3), axes=(0,2), reshape=False, order=0, mode='nearest')
            ndimage.rotate(image.copy(), 90 * random.randint(0,3), axes=(0,1), reshape=False, order=0, mode='nearest')
            ndimage.rotate(image.copy(), 90 * random.randint(0,3), axes=(1,2), reshape=False, order=0, mode='nearest')
    
        if strategy != 'v2':
            flip_axis = random.randint(0,3) # 3 plane flips + no flip
            if flip_axis == 0:
                 augmented_image = augmented_image[:, :, ::-1]
            elif flip_axis == 1:
                 augmented_image = augmented_image[:, ::-1, :]
            elif flip_axis == 2:
                 augmented_image = augmented_image[::-1, :, :]
    
            if strategy == 'v4' or strategy == 'v6':
                # Translation
                trans = (2 * (random.randint(-1,1)), 2 * (random.randint(-1,1)), 2 * (random.randint(-1,1)))
 
    return extract_patch_from_bigger_processed_lesion(augmented_image, trans, used_patch_size)

    
def extract_patch_from_bigger_processed_lesion(lesion, translation_vector, goal_patch_size):
    '''
    This function gets as input one lesion with multiple channels (phase, flair, mask, t2...)
    of size (channels, x, y, z) and it crops the lesion to "goal_patch_size" in all channels.
    If "translation_vector" specified, the crop is performed displaced by it first.
    '''
    goal_patch_size = np.array(goal_patch_size)
    mov = np.array(translation_vector)

    min_ = (lesion.shape[:-1] - goal_patch_size) // 2 + mov
    max_ = min_ + goal_patch_size
    
    assert (min_ >= (0,0,0)).all() and (max_ <= lesion.shape[:-1]).all()
    return lesion[min_[0]:max_[0], min_[1]:max_[1], min_[2]:max_[2], :]

