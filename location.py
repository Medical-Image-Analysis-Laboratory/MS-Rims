from tqdm import tqdm
import pandas as pd
import nibabel as nib
import json
import multiprocessing as mp
from scipy.ndimage import binary_dilation
from scipy.ndimage.morphology import distance_transform_edt

from config import *
from utils import read_patients_metadata

edts = {}
atlas = None

def compute_location_matches_patient(db, patient, replace = False):
    df_labels = pd.read_csv(PATH_ATLAS_LABELS, sep=";").sort_values("priority")
    labels = list(df_labels["label"])[:-1]
    
    #print(f"[START] Db {db}, pat {patient}.")
    dataset = AVAILABLE_DATASETS[db]
    pipeline = DERIVATIVES["LESIONS"]["pipeline"].format(PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2])
    
    try:
        json_lesions_path = dataset.get(return_type="filename", subject=f"{patient:03d}", scope=pipeline, extension='json')[0]
        mask_path = dataset.get(return_type="filename", subject=f"{patient:03d}", **CONTRASTS["MASK_MNI"])[0]
    except Exception:
        print(f"[ERROR] Patient {patient} skipped: file missing.")
        return

    mask = nib.load(mask_path).get_fdata().astype(int)

    # READ
    with open(json_lesions_path) as inp:
        pat_metadata = json.load(inp)

    #print("[RUN] Metadata loaded")
    for les in pat_metadata:
        if not replace and "location" in pat_metadata[les]:
            #print(f"[RUN] Lesion {les} skipped.")
            continue
        min_distance, min_label = float('inf'), None
        les = int(les)
        for label in labels:
            current = edts[label].copy()
            overlap = current[mask == les]
            if overlap.shape[0] == 0:
                print(f"[ERROR] No overlap with {df_labels.loc[df_labels['label'] == label, 'name'].array[0]} in lesion: {db} - {patient} - {les}")
                continue
            distance = np.min(overlap)
                
            overlap_needed = df_labels.loc[df_labels['label'] == label, 'overlap'].array[0]
            if distance < min_distance:
                if overlap_needed > 0 and len(np.where(overlap == 0)[0]) < overlap_needed:
                    continue # overlapping requirement not satisfied
                #elif overlap_needed > 0:
                    #print(f"Overlap enough with GM: {len(np.where(overlap == 0)[0])}")
                
                if distance <= df_labels.loc[df_labels['label'] == label, 'margin'].array[0]:
                    min_label = label
                    min_distance = distance
                    break # we have a priority match
        # no priority match => WM
        if min_label == None:
            min_label = 0 # WM
            min_distance = 0
        #print(f"{patient} - {les} - {df_labels.loc[df_labels['label'] == min_label, 'name'].array[0]} - {min_distance}")
        pat_metadata[str(les)]["location"] = df_labels.loc[df_labels['label'] == min_label, 'name'].array[0]
        pat_metadata[str(les)]["location_distance"] = min_distance

    os.remove(json_lesions_path)
    with open(json_lesions_path, "w") as outfile:
        json.dump(pat_metadata, outfile)
    return db, patient


def compute_locations(datasets = None, cpus = 6, replace = False):
    if datasets is None:
        datasets = list(range(len(AVAILABLE_DATASETS)))
        
    df_labels = pd.read_csv(PATH_ATLAS_LABELS, sep=";").sort_values("priority")
    labels = list(df_labels["label"])[:-1]
        
    global atlas
    atlas = nib.load(PATH_ATLAS).get_fdata().astype(int)

    # computations of edt for each label
    print("[LOG] Computing EDT matrices...")
    global edts
    edts = {}
    for i in tqdm(range(len(df_labels.index))):
        label = df_labels.iloc[i]["label"]
        edts[label] = distance_transform_edt(atlas != label)
    print("[LOG] EDT matrices computed.")
        
    pool = mp.Pool(cpus)
    processes = []
    for db in datasets:
        dataset = AVAILABLE_DATASETS[db]
        for patient in dataset.get_subjects():
            def callback(result):
                if result is not None:
                    print(f"[END] Patient {result[0]} - {result[1]} finished...")
            processes.append(pool.apply_async(compute_location_matches_patient, args=(db, int(patient)), callback=callback))
    for p in processes:
        p.get()
    pool.close()
    pool.join()
    