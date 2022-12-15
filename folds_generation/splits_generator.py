import sys
import os
import random
import numpy as np
import math
import logging

sys.path.append("..")
from utils import read_patients_metadata
from config import *

SEED = -1
TOTAL_FOLDS = 4
TOTAL_PATIENTS = 0
PERCENTAGES = (0.25, 0.25, 0.25, 0.25)
PATS_COUNTS = None
CENTER_PATIENTS = None
POS_PERF = 0
NEG_PERF = 0

POPULATION_SIZE = 100
MUTATION_PROBABILITY_BEST = 0.05
NUMBER_OF_CHANGES = 5
MAX_NUM_GENERATIONS = 100000
PROPORTION_SELECTION = 0.5
SAVE_GENERATIONS = 10
RIM_POS_IMPORTANCE = 3
MIN_VALUE_TO_SAVE = 500

DATASET_NAME = "DATASET_CHUV"
DATASET_ID = DATASET_CHUV_ID
DATASET = DATASET_CHUV

LOG = None


def loop():
    LAST_FILE = None
    LAST_BEST = MIN_VALUE_TO_SAVE
    FOLDER_TO_SAVE = f"./{DATASET_NAME}"
    
    if not os.path.exists(FOLDER_TO_SAVE):
        os.makedirs(FOLDER_TO_SAVE)
    
    last_min_value = float('inf')
    popu = generate_population()
    #print(sorted(int(get_fitness_score(idv)[0]) for idv in popu))
    for i in range(MAX_NUM_GENERATIONS):
        popu, scores, _, _ = selection(popu)
        to_generate = POPULATION_SIZE - len(popu)
        
        for j in range(to_generate):
            father_id = np.random.randint(0, len(popu))
            mother_id = father_id
            while mother_id == father_id:
                mother_id = np.random.randint(0, len(popu))
                
            father = popu[father_id]
            mother = popu[mother_id]
            son = crossover(father, mother)
            popu.append(son)
            
        max_score = np.max(scores)
        min_score = np.min(scores)
        for j in range(len(popu)):
            mutate(popu[j], min_score, max_score)
            
        if i % SAVE_GENERATIONS == 0:
            best, _, min_value, mean_value = selection(popu, number=1)
            logging.info(f"{i} - {int(min_value)} - {int(mean_value)}")
            if min_value < MIN_VALUE_TO_SAVE and min_value < LAST_BEST:
                path = os.path.join(FOLDER_TO_SAVE, f"{int(min_value)}_{SEED}_{i}")
                np.save(path, best)
                if LAST_FILE is not None:
                    os.remove(LAST_FILE)
                LAST_FILE = path + ".npy"
                LAST_BEST = min_value
        

def run(db_id, seed, log = None):
    if not os.path.exists("./LOGS"):
        os.makedirs("./LOGS")
    logging.basicConfig(filename=f'./LOGS/{db_id}_{seed}.txt', level=logging.INFO)
    
    global SEED
    SEED = seed
    global DATASET_NAME
    global DATASET_ID
    global DATASET
    if db_id == DATASET_CHUV_ID:
        DATASET_NAME = "DATASET_CHUV"
        DATASET_ID = DATASET_CHUV_ID
        DATASET = DATASET_CHUV
    elif db_id == DATASET_BASEL_ID:
        DATASET_NAME = "DATASET_BASEL"
        DATASET_ID = DATASET_BASEL_ID
        DATASET = DATASET_BASEL
    else:
        print("[ERROR] Dataset ID not valid.")
        return -1
    
    metadata_original = read_patients_metadata(PATCH_SIZE)[DATASET_ID]

    volumes = [int(metadata_original[pat][les]["volume"]) 
               for db in metadata_original 
               for pat in metadata_original 
               for les in metadata_original[pat]]

    pos_to_subj = {}
    subj_to_pos = {}
    counter = 0

    global CENTER_PATIENTS
    CENTER_PATIENTS = []
    for pat in DATASET.get_subjects():
        pos_to_subj[counter] = pat
        subj_to_pos[pat] = counter
        CENTER_PATIENTS.append(counter)
        counter += 1

    metadata = {}
    for pat_int in list(metadata_original.keys()):
        pat = f"{pat_int:03d}"
        metadata[subj_to_pos[pat]] = metadata_original[pat_int]
        for les in list(metadata[subj_to_pos[pat]].keys()):
            if bool(metadata[subj_to_pos[pat]][les]["ignore"]):
                metadata[subj_to_pos[pat]].pop(les, None)

    global PATS_COUNTS
    PATS_COUNTS = np.zeros([len(subj_to_pos), 2], dtype='int32')

    for pat in CENTER_PATIENTS:
        patient_rims_count = len([les for les in metadata[pat].keys() if int(les) // 1000 == 1])
        patient_non_rims_count = len([les for les in metadata[pat].keys() if int(les) // 1000 == 2])
        PATS_COUNTS[pat][0] = patient_rims_count
        PATS_COUNTS[pat][1] = patient_non_rims_count

    logging.info(f"{np.sum(PATS_COUNTS, axis=0)}")
    global TOTAL_PATIENTS
    TOTAL_PATIENTS = PATS_COUNTS.shape[0]
    
    pos, neg = np.sum(PATS_COUNTS[CENTER_PATIENTS], axis=0)
    global POS_PERF
    global NEG_PERF
    POS_PERF = pos
    NEG_PERF = neg
    
    loop()

def generate_individual(p = None):
    if p is None:
        p = np.ones(TOTAL_FOLDS) / TOTAL_FOLDS
        
    basic = np.zeros([TOTAL_FOLDS, TOTAL_PATIENTS]).astype(bool)
    for i in range(TOTAL_PATIENTS):
        idx = np.random.choice(TOTAL_FOLDS, p = p)#np.random.randint(0, TOTAL_FOLDS)
        basic[idx][i] = True
    return basic
    
def generate_population():
    return [generate_individual() for i in range(POPULATION_SIZE)]

# O(n) complexity, cool!
def get_fitness_score(idv):
    errors = []
    abs_error = 0
    for i in range(idv.shape[0]):
        pos_i, neg_i = np.sum(PATS_COUNTS[np.intersect1d(np.where(idv[i])[0], CENTER_PATIENTS)], axis=0)
        error_pos = pos_i - POS_PERF * PERCENTAGES[i]
        error_neg = neg_i - NEG_PERF * PERCENTAGES[i]
        
        abs_error += abs(error_pos) * 10 * RIM_POS_IMPORTANCE + abs(error_neg)
        errors.append(error_pos + error_neg)
        
    return abs_error, errors

def is_valid(result):
    # 1. Every patient 1 fold assigned
    # 2. One patient cannot have more than 1 fold assigned
    assert result.shape[1] == TOTAL_PATIENTS and result.shape[0] == TOTAL_FOLDS
    return (np.sum(result.astype(int), axis=0) == np.ones(result.shape[1])).all()

# selects best "number" individuals from population
def selection(population, number=None):
    if number == None:
        number = int(len(population) * PROPORTION_SELECTION)
        
    scores_popu = np.array([get_fitness_score(idv)[0] for idv in population])
    min_value = np.min(scores_popu)
    mean_value = np.mean(scores_popu)
    scores_sel = []
    selected_popu = []
    for i in range(number):
        # get the max and put the max to -inf so it is not chosen again
        idx = np.argmin(scores_popu)
        selected_popu.append(population[idx])
        scores_sel.append(scores_popu[idx])
        scores_popu[idx] = float('inf')
        
    return selected_popu, scores_sel, min_value, mean_value

def crossover(idv1, idv2):
    if((idv1 == idv2).all()):
        print("EQUAL!")
    # we choose the 2 best splits of the father and mother for the son
    if random.random() < 0.5:
        # to make it gender equally
        aux = idv1
        idv1 = idv2
        idv2 = aux
    
    idv1_errors = np.abs(get_fitness_score(idv1)[1])
    idv1_probs = np.max(idv1_errors) - idv1_errors # the lower error, the higher prob
    idv1_probs = idv1_probs / np.sum(idv1_probs)
    if np.isnan(idv1_probs).any():
        return generate_individual()
    idv1_best = np.random.choice(TOTAL_FOLDS, p=idv1_probs)
    idv1_patients_to_keep = list(np.where(idv1[idv1_best])[0])
    
    idv2_errors = np.abs(get_fitness_score(idv2)[1])
    idv2_probs = np.max(idv2_errors) - idv2_errors # the lower error, the higher prob
    idv2_probs[idv1_best] = 0
    idv2_probs = idv2_probs / np.sum(idv2_probs)
    if np.isnan(idv2_probs).any():
        return generate_individual()
    idv2_best = np.random.choice(TOTAL_FOLDS, p=idv2_probs)
    idv2_patients_to_keep = list(np.where(idv2[idv2_best])[0])
    
    probs = np.ones(TOTAL_FOLDS)
    probs[idv1_best] = 0
    probs[idv2_best] = 0
    probs = probs / np.sum(probs)
    son = generate_individual(p = probs)
    
    son[:, idv1_patients_to_keep] = idv1[:, idv1_patients_to_keep]
    son[:, idv2_patients_to_keep] = idv2[:, idv2_patients_to_keep]
    
    return son

def normalize(v):
    return (v - np.min(v)) / (np.max(v) - np.min(v))

def mutate(idv, min_value, max_value):
    score, errors = get_fitness_score(idv)
    mid_value = min_value + (max_value - min_value) / 2
    k = -math.log((1 - MUTATION_PROBABILITY_BEST) / MUTATION_PROBABILITY_BEST) / (min_value - mid_value)
    sigmoid = 1 / (1 + math.exp(-k * (score - mid_value)))
    if random.random() < sigmoid:
        for i in range(NUMBER_OF_CHANGES):
            errors = get_fitness_score(idv)[1]
            
            # First que choose the fold of the patient to mute
            probs_f = normalize([max(0, e) for e in errors]) 
            probs_f = 0.1 + 0.8 * probs_f
            probs_f = probs_f / np.sum(probs_f)
            #print(probs_f)
            if np.isnan(probs_f).any():
                return
            fold_to_mute_from = np.random.choice(TOTAL_FOLDS, p=probs_f)
            
            # Second que choose the fold of the patient to mute
            probs_t = normalize([max(0, -e) for e in errors]) 
            probs_t[fold_to_mute_from] = 0
            probs_t = 0.1 + 0.8 * probs_t
            probs_t = probs_t / np.sum(probs_t) # sum = 1
            #print(probs_t)
            if np.isnan(probs_t).any():
                return
            fold_to_mute_to = np.random.choice(TOTAL_FOLDS, p=probs_t) 
            
            # Random patient inside the fold FROM
            candidates = np.where(idv[fold_to_mute_from] == True)[0]
            if len(candidates) > 0:
                patient_from = np.random.choice(candidates)
                #print(f"[{patient_from}] {fold_to_mute_from} -> {fold_to_mute_to}")
                idv[fold_to_mute_from][patient_from] = False
                idv[fold_to_mute_to][patient_from] = True
            
        #print("END")
            #patient_to_mute = np.random.randint(0, TOTAL_PATIENTS)
            #idv[:, patient_to_mute] = np.zeros(TOTAL_FOLDS)
            #idv[np.random.randint(0, TOTAL_FOLDS), patient_to_mute] = True