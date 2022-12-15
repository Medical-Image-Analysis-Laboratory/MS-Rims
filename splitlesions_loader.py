import json
import os
import numpy as np
from nibabel import load as load_nii
import nibabel as nib
from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure as gbs
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import sys
sys.path.append("..")
from config import *
from munkres import Munkres
import cv2
import pandas as pd
from deformations import deform_and_crop_lesion
from utils_basic import crop_lesions_from_image

    
    
def load_patient_split_lesions(dataset_id, pat_id, patch_size, version, deformed, only_real):
    '''
        Load all lesions of the patient "pat_id" from dataset "dataset_id".
        Deformed versions are generated on the fly.
    '''
        
    dataset = AVAILABLE_DATASETS[dataset_id]
    lesions = {}
    lesions_patches = {}
    contrasts_not_found = []
    
    path_metadata = dataset.get(return_type="filename", subject=f"{int(pat_id):03d}", scope=SPLIT_LESIONS_METADATA[version]["pipeline"], suffix=SPLIT_LESIONS_METADATA[version]["suffix"], acquisition=None, extension="csv")
    if len(path_metadata) != 1:
        #print(f"ERROR: {dataset_id} - {pat_id}")
        return dataset_id, pat_id
    
    for c in PURE_CONTRASTS:
        sf, acq, sc, ext = CONTRASTS[c]["suffix"], CONTRASTS[c]["acquisition"], CONTRASTS[c]["scope"], CONTRASTS[c]["extension"]
        try:
            contrast_fn = dataset.get(return_type="filename", subject=f"{pat_id:03d}", scope=sc, suffix=sf, acquisition=acq, extension=ext)[0]
        except IndexError:
            contrasts_not_found.append(c)
            continue
            
        image = nib.load(contrast_fn).get_fdata()
            
        df = pd.read_csv(path_metadata[0])
        if only_real:
            # we only load the real ones
            df = df[df["real"]]
            
        lesions_meta = {}
        for index, row in df.iterrows():
            lesions_meta[row["lesion"]] = {"center": (row["x"], row["y"], row["z"])}
        
        if deformed is None:
            patches = crop_lesions_from_image(image, lesions_meta, patch_size, is_mask = (c == "MASK"))
        else:
            patches = crop_lesions_from_image(image, lesions_meta, PATCH_SIZE_DEFORMATIONS, is_mask = (c == "MASK"))
            
        # conversion to friendly format
        for les in patches.keys():
            if int(les) not in lesions_patches.keys():
                lesions_patches[int(les)] = {}
            lesions_patches[int(les)][c] = patches[les]
        
    if deformed:
        # Lesion seed: int(AXYYYZZZZ) where A kernel, X dataset, YYY patient, ZZZZ lesion
        for les_id in lesions_patches.keys():
            lesion_seed = int(f"{deformed}{dataset_id}{pat_id:03d}{les_id}")
            np.random.seed(lesion_seed)
            lesions_patches[int(les_id)] = deform_and_crop_lesion(les_id, lesions_patches[les_id], patch_size, recenter=False)
            
    if len(contrasts_not_found) > 0:
        #print(f"[ERROR_{dataset_id}_{pat_id}] Contrasts not found: {contrasts_not_found}")
        pass
    return dataset_id, pat_id, lesions_patches, lesions_meta
