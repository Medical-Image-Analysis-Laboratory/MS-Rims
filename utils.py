import os

import numpy as np
import scipy.ndimage as snd
import nibabel as nib
import math
import time
from config import *
from tqdm import tqdm
from scipy import ndimage
import random
import cv2
import multiprocessing as mp
import json
from skimage.measure import regionprops
import pandas as pd
from sklearn import metrics

from splitlesions_loader import load_patient_split_lesions
from deformations import deform_and_crop_lesion
from utils_basic import crop_lesions_from_image

from bids import BIDSLayout
from bids.analysis import Analysis


def generate_and_get_folder_for_lesions(mask_path, patch_size, subject, *, session=1, deformed=None):
    """
        Function that generates the derivatives folder according to BIDS structure to
        save the extracted lesions from all the sequences. It also creates the jsons
        needed and the folders.
        

            folder: path to the folder where files will be stored.
                    e.g. /GERMAN/datasets/BASEL_DATASET/derivatives/lesions_34_34_34/sub-001/ses-01/
    """
    dataset_name = mask_path.replace(PATH_DATA, "").split("/")[0]
    der_folder_path = os.path.join(PATH_DATA, dataset_name, "derivatives")
    if deformed is None:
        extension = DERIVATIVES["LESIONS"]["extension"]
        der_folder_name = DERIVATIVES["LESIONS"]["folder_name"].format(patch_size[0], patch_size[1], patch_size[2])
        pipeline = DERIVATIVES["LESIONS"]["pipeline"].format(patch_size[0], patch_size[1], patch_size[2])
    else:
        extension = DERIVATIVES["LESIONS_DEF"]["extension"]
        der_folder_name = DERIVATIVES["LESIONS_DEF"]["folder_name"].format(patch_size[0], patch_size[1], patch_size[2], deformed)
        pipeline = DERIVATIVES["LESIONS_DEF"]["pipeline"].format(patch_size[0], patch_size[1], patch_size[2], deformed)

    lesions_folder = os.path.join(der_folder_path, der_folder_name)
    
    # we create folder if it does not exist
    if not os.path.exists(lesions_folder):
        try:
            os.makedirs(lesions_folder)
            print(f"[INFO] Derivatives folder for '{der_folder_name}' successfully created.")
        except: # Sometimes in multiprocessing this check is true for several processes and crashes
            pass
        
    # we create the description of the derivatives if it does not exist
    dataset_description_path = os.path.join(lesions_folder, "dataset_description.json")
    if not os.path.exists(os.path.join(dataset_description_path)):
        descriptor = {
            "Name": der_folder_name,
            "BIDSVersion": BIDS_VERSION,
            "PipelineDescription": {
                "Name": pipeline
            }
        }
        with open(dataset_description_path, "w") as outfile:
            json.dump(descriptor, outfile)
        print(f"[INFO] Description file for '{der_folder_name}' successfully created.")
    
    # we create the path for the generated file
    folder = os.path.join(lesions_folder, f"sub-{subject:03d}", f"ses-{session:02d}")
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder, extension

def __check_to_clean(lesion, metadata, pat_id, lesion_id, goal_patch_size=PATCH_SIZE_TRAINING):
    '''
        Checks if lesion has to be removed according strategy "v2". Need to specify the patch size with which the model will work.
        RETURNS:
            - Boolean, True if need to be removed.
    '''
    if WORKING_MODE_SKULL_STRIPPED:
        print("[ERROR] Can't work in skull stripped mode.")
        return
    RIM_INTRUSION_ALLOWED = 4 #%
    THRESHOLD_AIR_ARTIFACT = 0.10 #per 1
    MIN_VOLUME, MAX_VOLUME = 45, 10000
    reasons = []
    reasons_extended = ""
    
    # CONDITION 0-1: volume
    if int(metadata["volume"]) < MIN_VOLUME:
        reasons.append(EXCLUSION_SMALL)
        reasons_extended += f"[Volume lower than accepted threshold ({MIN_VOLUME})] "
    elif int(metadata["volume"]) > 10000:
        reasons.append(EXCLUSION_BIG)
        reasons_extended += f"[Volume higher than accepted threshold ({MAX_VOLUME})] "
    
    if "MASK" not in lesion.keys():
        print(pat_id)
        print(lesion_id)
    shape = lesion["MASK"].shape
    goal_patch_size = np.array(goal_patch_size)
    
    min_ = (shape - goal_patch_size) // 2
    max_ = min_ + goal_patch_size
    assert (min_ >= (0,0,0)).all() and (max_ <= shape).all()
    
    lesion_cropped = {} # dict of contrasts
    for c in lesion.keys():
        lesion_cropped[c] = lesion[c][min_[0]:max_[0], min_[1]:max_[1], min_[2]:max_[2]]

    total_volume = (goal_patch_size[0] * goal_patch_size[1] * goal_patch_size[2])
    # CONDITION 2: rim infiltration
    if lesion_id // 2000 == 1:
        rim_infiltration_vol = np.sum((lesion_cropped["MASK"] // 1000 == 1).astype(int))
        rim_inf_percent = 100 * rim_infiltration_vol / total_volume

        if rim_inf_percent >= RIM_INTRUSION_ALLOWED:
            reasons.append(EXCLUSION_RIM_INTRUSION)
            reasons_extended += f"[Rim intrusion of {rim_inf_percent} > {RIM_INTRUSION_ALLOWED}] "
    
    # CONDITION 3: air artifacts
    phase = lesion_cropped["T2STAR_PHASE"]
    phase_normalized = (phase - np.min(phase))/(np.max(phase) - np.min(phase))
    if np.sum((phase_normalized < THRESHOLD_AIR_ARTIFACT).astype(int)) >= total_volume / 28:
        # Lesion with air artifact in PHASE.
        reasons.append(EXCLUSION_AIR_ARTIFACT)
        reasons_extended += "[Lesion with air artifact in PHASE] "
        
    return len(reasons) != 0, reasons, reasons_extended

def apply_cleaner(patch_size: np.array):
    if WORKING_MODE_SKULL_STRIPPED:
        print("[ERROR] Can't work in skull stripped mode.")
        return
    lesions = load_lesions(patch_size, only_cleaned=False)
    for db_id in range(len(AVAILABLE_DATASETS)):
        dataset = AVAILABLE_DATASETS[db_id]
        for pat in dataset.get_subjects():
            subject = int(pat)
    
            pipeline = PATIENTS_METADATA["pipeline"].format(patch_size[0], patch_size[1], patch_size[2])
            # We retrieve the lesions of the patient
            json_lesions_path = dataset.get(return_type="filename", subject=f"{subject:03d}", scope=PATIENTS_METADATA["pipeline"].format(patch_size[0], patch_size[1], patch_size[2]), extension='json')[0]
            
            if os.path.exists(json_lesions_path):
                # READ
                with open(json_lesions_path) as inp:
                    pat_metadata = json.load(inp)
                  
                # UPDATE
                for les_id in pat_metadata.keys():
                    if int(les_id) in lesions[db_id][subject]:
                        if "MASK" not in lesions[db_id][subject][int(les_id)]:
                            print(les_id)
                        to_ignore, reason, comment = __check_to_clean(lesions[db_id][subject][int(les_id)], pat_metadata[les_id], subject, int(les_id))
                        pat_metadata[les_id]["ignore"] = to_ignore
                        pat_metadata[les_id]["reasons_to_ignore"] = reason
                        pat_metadata[les_id]["reasons_to_ignore_ext"] = comment
                    else:
                        print(pat_metadata[les_id])
                        pat_metadata[les_id]["ignore"] = True
                        pat_metadata[les_id]["reasons_to_ignore"] = [-1, ]
                        pat_metadata[les_id]["reasons_to_ignore_ext"] = "[Bbox outside image boundaries.] "
                    
                # SAVE
                os.remove(json_lesions_path)
                with open(json_lesions_path, "w") as outfile:
                    json.dump(pat_metadata, outfile)
                # tota la lògica aqui
                # IMPORTANT: modificar dataset_description amb la strategy usada per netejar
                # que cada cop que es runegi el cleaner, es faci desde zero
                # Arreglar tambe les deformed versions perque hi hagi una flag de cleaner,
                # perque nomes es generaran les deformed versions dels que han passat el cleaner.
                # TODO: modificar el reader afegint una flag permetre que nomes llegeixi les lesions que 
                # hagin passat el cleaner.
            else:         
                print(f"[ERROR] Patient {pat} skipped: no json file found.")

def extract_lesions_from_mask(mask):#, filter_by_vol = True):
    """
    Python function to extract the center of lesions and their volume from the mask of the patient (segmentation).
    If activated, lesions are filtered by volume.
    
    Args:
        mask: numpy array containing the lesion mask

    Returns:
        lesions: dictionary where KEY is the ID of the lesion, and the value is a dictionary:
                {
                    "rim_presence": True if rim positive, False if rim negative,
                    "center": array of 3 integers containing x, y, z of its center of mass,
                    "volume": integer
                }
    """
    lesions = {}
    lesion_ids = np.unique(mask)[1:].astype(int)
    #print(lesion_ids)
    
    for current_id in lesion_ids:
        current_mask =  (mask == current_id).astype(int)

        # Compute the volume
        vol = np.sum(current_mask)
        
        # FALSE rn!! Only patches having their limits within the volume and within values of volume
        #if not filter_by_vol or (vol > 100 and vol < 15000):  
        com = np.round(snd.measurements.center_of_mass(current_mask))
        
        prop = regionprops((mask == current_id).astype(int))[0]
        x0, y0, z0, x, y, z = prop.bbox
        
        lesions[str(current_id)] = {
            "rim_presence": str(current_id // 1000 == 1),
            "ignore": False,
            "reasons_to_ignore": "",
            "reasons_to_ignore_ext": "",
            "center": com.astype('int32').tolist(),
            "volume": str(vol),
            "bbox": ((x0, y0, z0), (x, y, z))
        }

    return lesions

def extract_lesions_from_patient(dataset_id, subject, patch_size: np.array, regenerate=False, only_metadata=False):
    """
        Function that extracts the lesions from all contrasts of one patient, specified by parameter,
        and saves the result as "dat" files in the "derivatives" folder. Lesions are stored in a different
        file for EACH contrast.
    """
    if WORKING_MODE_SKULL_STRIPPED:
        print("[ERROR] Can't work in skull stripped mode.")
        return
    
    dataset = AVAILABLE_DATASETS[dataset_id]
    
    mask_fn = dataset.get(return_type="filename", subject=f"{subject:03d}", **CONTRASTS["MASK"])[0]
    lesions_folder_path, lesions_ext = generate_and_get_folder_for_lesions(mask_fn, patch_size, subject)
    json_lesions_path = os.path.join(lesions_folder_path, f"sub-{subject:03d}_ses-01.json")
    
    if os.path.exists(json_lesions_path):
        # Read lesions data from json file
        with open(json_lesions_path) as inp:
            lesions = json.load(inp)
    else:         
        # Compute lesions data from mask and save it to json file
        mask = nib.load(mask_fn).get_fdata()
        lesions = extract_lesions_from_mask(mask)
        # we update the json file in case new lesions are ignored
        with open(json_lesions_path, "w") as outfile:
            json.dump(lesions, outfile)
    
    if only_metadata:
        return
    
    # Extraction of lesions from other contrasts
    for c in PURE_CONTRASTS:
        sf, acq, sc, ext = CONTRASTS[c]["suffix"], CONTRASTS[c]["acquisition"], CONTRASTS[c]["scope"], CONTRASTS[c]["extension"]
        try:
            contrast_fn = dataset.get(return_type="filename", subject=f"{subject:03d}", scope=sc, suffix=sf, acquisition=acq, extension=ext)[0]
        except IndexError:
            print(f"[ERROR] Contrast not found: {subject} - {c}")
            continue
            
        filename = contrast_fn.split("/")[-1].replace("nii.gz", lesions_ext)
        final_path = os.path.join(lesions_folder_path, filename)
        
        if os.path.exists(final_path) and not regenerate:
            continue
            
        image = nib.load(contrast_fn).get_fdata()
        
        patches = crop_lesions_from_image(image, lesions, patch_size, is_mask = (c == "MASK"))
        
        __save_patches(final_path, patches)

def __save_patches(lesions_filepath, lesions):
    '''
        Save (to the specified file) all the lesions of a patient passed as a dictionary 
        with KEY the lesion_id and VALUE the numpy array of one contrast.
    '''
    if not os.path.exists(os.path.dirname(lesions_filepath)):
        os.makedirs(os.path.dirname(lesions_filepath))
    with open(lesions_filepath, "w") as lesions_file:
        for lesion_id in sorted(lesions.keys()):
            patch_array = lesions[lesion_id].flatten("C")
            # write it in the file
            lesions_file.write(f"{lesion_id}")
            for v in patch_array:
                lesions_file.write(f" {v}")
            lesions_file.write("\n")

def __read_patches(lesions_filepath, patch_size):
    '''
        Load all lesions from processed file ".dat". Each file contains ONE contrast.
        
        - RETURN: dictionary where you can access to lesions of the patient like
                    lpp[LESION_ID]
    '''
    lesions = {}
    with open(lesions_filepath) as lesions_file:
  
      for line in lesions_file.read().splitlines():
            #if not line.strip():
                #continue
            splitted = line.split(" ")
            lesion_id = int(splitted[0])
            lesions[lesion_id] = np.array(splitted[1:], dtype='float32').reshape(patch_size[0], patch_size[1], patch_size[2])
     
    return lesions

def extract_lesions_from_source_data(patch_size: np.array, *, regenerate=False, cpus=32, asyncr=True, only_metadata=False):
    patch_size = np.array(patch_size)
    
    print("[START] Extracting lesions...")
    start = time.time()
    pool = mp.Pool(min(cpus, mp.cpu_count()))
    processes = []
    try:
        for dataset_id in range(len(AVAILABLE_DATASETS)):  
            dataset = AVAILABLE_DATASETS[dataset_id]
            for subject in dataset.get_subjects():
                if asyncr:
                    processes.append(pool.apply_async(extract_lesions_from_patient, args=(dataset_id, int(subject), patch_size, regenerate, only_metadata)))
                else:
                    extract_lesions_from_patient(dataset_id, int(subject), patch_size, regenerate, only_metadata)
    except:
        print("[ERROR] One process of the pool failed. Terminating...")
        pool.terminate()
        
    for p in processes:
        p.get()
    pool.close()
    pool.join()
    print("[END] Extraction of lesions ended successfully!")
    print(f"[INFO] It took {(time.time() - start) / 60:.2f} minutes to check/extract the data.")

# TO RENEW
def get_lesion(patient, lesion, patch_size = [28,28,28], deformed=None):
    '''
        Returns a lesion by (patient, lesion) without the need of loading all data.
    '''
    path = os.path.join(PATH_DATA, str(patient))
    return load_data_from_dir(path, patch_size, deformed=deformed)[lesion]

def read_patients_metadata(patch_size):
    '''
        Load all metadata of the lesions of the patient "pat_id" from dataset "dataset_id".
        - RETURN: dictionary where you can access to the metadata of any lesion.
    '''
    metadata = {}
    for i in range(len(AVAILABLE_DATASETS)):
        metadata[i] = {}
        dataset = AVAILABLE_DATASETS[i]
        for pat in dataset.get_subjects():
            pat_id = int(pat)
            # Establish the folder inside derivatives where we will search
            pipeline = PATIENTS_METADATA["pipeline"].format(patch_size[0], patch_size[1], patch_size[2])

            # We retrieve the lesions of the patient
            files = dataset.get(return_type="filename", subject=f"{pat_id:03d}", scope=PATIENTS_METADATA["pipeline"].format(patch_size[0], patch_size[1], patch_size[2]), extension='json')
            if len(files) != 1:
                print(f"[WARNING] Subject {pat_id} was omitted because no lesions 'json' file was found.")
                continue
            with open(files[0]) as inputfile:
                metadata[i][pat_id] = json.load(inputfile)
    return metadata

def get_dataframe_from_metadata(split_version = None):
    # We process the data
    data = read_patients_metadata(PATCH_SIZE)
    
    pd_data = ((db, pat, int(les), int(les) // 1000 == 1, int(data[db][int(pat)][les]["volume"]), data[db][int(pat)][les]["ignore"],
                np.NaN if len(data[db][int(pat)][les]["reasons_to_ignore"]) == 0 else ', '.join([get_exclusion_reason(reason) for reason in data[db][int(pat)][les]["reasons_to_ignore"]]),
                ', '.join([str(i) for i in data[db][int(pat)][les]["reasons_to_ignore"]]),
               data[db][pat][les]["location"] if "location" in data[db][pat][les] else np.NaN,
               (data[db][int(pat)][les]["bbox"][1][0] - data[db][int(pat)][les]["bbox"][0][0], 
                data[db][int(pat)][les]["bbox"][1][1] - data[db][int(pat)][les]["bbox"][0][1], 
                data[db][int(pat)][les]["bbox"][1][2] - data[db][int(pat)][les]["bbox"][0][2]))
               for db in data
               for pat in data[db]
               for les in data[db][pat])
    
    return pd.DataFrame(data = pd_data, columns=["dataset", "patient", "lesion", "rim", "volume", "ignore", "main_reason", "reasons", "location", "size"])

def get_dataframe_from_split_lesions(version):
    to_concat = []
    for dataset_id in range(len(AVAILABLE_DATASETS)):
        dataset = AVAILABLE_DATASETS[dataset_id]
        for pat in dataset.get_subjects():
            paths = dataset.get(return_type="filename", subject=f"{pat}", scope=SPLIT_LESIONS_METADATA[version]["pipeline"], suffix=SPLIT_LESIONS_METADATA[version]["suffix"], acquisition=None, extension="csv")
            if len(paths) == 1:
                to_concat.append(pd.read_csv(paths[0]))
                #pd.read_csv(paths[0])[["dataset_id", "patient", "lesion", "x", "y", "z", "percentage_rims", "voxels_rims", "real"]].to_csv(paths[0], index=False)

    df = pd.concat(to_concat)
    return df

def load_patient_lesions(dataset_id, pat_id, patch_size, deformed, only_cleaned=True):
    '''
        Load all lesions of the patient "pat_id" from dataset "dataset_id".
    '''
    
    # Establish the folder inside derivatives where we will search
    if deformed is None:
        extension = DERIVATIVES["LESIONS"]["extension"]
        pipeline = DERIVATIVES["LESIONS"]["pipeline"].format(patch_size[0], patch_size[1], patch_size[2])
    else:
        extension = DERIVATIVES["LESIONS_DEF"]["extension"]
        pipeline = DERIVATIVES["LESIONS_DEF"]["pipeline"].format(patch_size[0], patch_size[1], patch_size[2], deformed)
        
    dataset = AVAILABLE_DATASETS[dataset_id]
    lesions = {}
    contrasts_not_found = []
    # We retrieve the lesions of the patient
    files = dataset.get(return_type="filename", subject=f"{pat_id:03d}", scope=PATIENTS_METADATA["pipeline"].format(patch_size[0], patch_size[1], patch_size[2]), extension='json')
    if len(files) != 1:
        print(f"[WARNING] Subject {pat_id} was omitted because no lesions 'json' file was found.")
        return dataset_id, pat_id
    #print(files[0])
    with open(files[0]) as inputfile:
        json_metadata = json.load(inputfile)
        lesion_keys = np.array(list(json_metadata.keys())).astype(int)
    for les_id in lesion_keys:
        lesions[les_id] = {}

    # We extract the lesions for each contrast
    for contrast in PURE_CONTRASTS:
        
        # Search for the lesions extracted from the specified contrast
        lp = dataset.get(return_type="filename", subject=f"{pat_id:03d}", scope=pipeline, suffix=CONTRASTS[contrast]["suffix"], acquisition=CONTRASTS[contrast]["acquisition"], extension=extension)

        if len(lp) == 0:
            #print(f"[WARNING] Lesions for contrast {contrast} for subject {pat_id} not extracted ({pipeline}).")
            contrasts_not_found.append(contrast)
            continue
            #return pat_id
        elif len(lp) > 1:
            raise Exception(f"[ERROR] More than 1 file found for the same contrast and subject: {pat_id}, {contrast}, {pipeline}")

        # We load it
        lp = lp[0]
        if os.path.isfile(lp):
            # precomputed lesions file exists so we load from that
            try:
                lesions_aux = __read_patches(lp, patch_size)
            except Exception:
                raise Exception(f"[ERROR] Broken 'dat' file: {pat_id}, {contrast}, {pipeline}")
                
            # we store the lesions read in our dictionary format. ONLY LESIONS IN THE JSON FILE CONSIDERED
            for les_id in lesion_keys:
                # 0 not in json_metadata[str(les_id)]["reasons_to_ignore"] BECAUSE WE NEVER WANT THEM, not even for testing
                try:
                    if les_id in lesions_aux.keys() and ((not only_cleaned and (str(0) not in json_metadata[str(les_id)]["reasons_to_ignore"])) or not bool(json_metadata[str(les_id)]["ignore"])):
                        lesions[les_id][contrast] = lesions_aux[les_id]
                    else:
                        lesions.pop(les_id, None)
                except:
                    print(f"METADATA_ERROR: {pat_id} - {les_id}")
                    
        else:
            raise Exception(f"[ERROR] ''.dat' file does not exist: {pat_id}, {contrast}, {pipeline}")
        
    if len(contrasts_not_found) > 0:
        print(f"[ERROR_{dataset_id}_{pat_id}] Contrasts not found: {contrasts_not_found}")
    return dataset_id, pat_id, lesions, json_metadata

def load_lesions(patch_size, *, deformed=None, only_cleaned=True, debug=False, cpus=32, asyncr = True, from_segmentation=False, only_real=True, split_version="v02"):
    patch_size = np.array(patch_size)
    '''
        Load all patients from all datasets available in "config.py".
        - RETURN: dictionary where you can access to any lesion and contrast like
                    lpp[DATASET_ID][PATIENT][LESION_ID][CONTRAST]
    '''
    cpus = min(cpus, mp.cpu_count()) if asyncr else 1
    print(f"[START] Loading lesions (deformed={deformed}, asyncr={asyncr}, cpus={cpus}, segm={from_segmentation})...")
    
    lpp = {}
    start = time.time()
    
    pool = mp.Pool(cpus)
    errors = []
    processes = []
    #try:
    for dataset_id in range(len(AVAILABLE_DATASETS)):
        dataset = AVAILABLE_DATASETS[dataset_id]
        lpp[dataset_id] = {}

        for pat in dataset.get_subjects():
            def callback(result):
                if result is not None and len(result) != 2: # int for errors
                    ds_id, sub_id, res, metadata = result
                    lpp[ds_id][int(sub_id)] = res
                else:
                    errors.append(result) #sometimes, print does not show because of "race condition"
            if asyncr:
                if from_segmentation:
                    processes.append(pool.apply_async(load_patient_split_lesions, args=(dataset_id, int(pat), patch_size, split_version, deformed, only_real), callback=callback))
                else:
                    processes.append(pool.apply_async(load_patient_lesions, args=(dataset_id, int(pat), patch_size, deformed, only_cleaned), callback=callback))
            else:
                if from_segmentation:
                    result = load_patient_split_lesions(dataset_id, int(pat), patch_size, split_version, deformed, only_real)
                else:
                    result = load_patient_lesions(dataset_id, int(pat), patch_size, deformed, only_cleaned)
                callback(result)
                
    #except Exception as ex:
    #    print(ex)
    #    print("[ERROR] One process of the pool failed. Terminating...")
    #    pool.terminate()
        
    if asyncr:
        for p in processes:
            p.get()
        pool.close()
        pool.join()
    if len(errors) > 0:
        print(f"[ERROR] Patients not loaded successfully: {len(errors)}, {sorted(errors)}")
    print("[END] Lesions loaded successfully!")
    print(f"[INFO] It took {(time.time() - start) / 60:.2f} minutes to load the data.")
    return lpp

#TO TEST
def copy_data(lpp):
    '''
        RETURN: Deep copy of a dictionary coming from "load_lesions".
    '''
    lpp_copy = {}
    for db_id in lpp.keys():
        lpp_copy[db_id] = {}
        for patient in lpp[db_id].keys():
            lpp_copy[db_id][patient] = {}
            for lesion in lpp[db_id][patient].keys():
                lpp_copy[db_id][patient][lesion] = {}
                for contrast in lpp[db_id][patient][lesion]:
                    lpp_copy[db_id][patient][lesion][contrast] = lpp[db_id][patient][lesion][contrast].copy()
    return lpp_copy
        
def normalize_patch(img_patch, normalize):
    if normalize == "local_max_old":
        img_patch = img_patch / np.max(img_patch)
    elif normalize == "local_max":
        img_patch = 2 * ((img_patch - np.min(img_patch))/(np.max(img_patch) - np.min(img_patch))) - 1
    elif normalize == "local_max_01":
        img_patch = (img_patch - np.min(img_patch))/(np.max(img_patch) - np.min(img_patch))
    elif normalize == "mean_std":
        img_patch = (img_patch - np.mean(img_patch)) / np.std(img_patch)
    else:
        raise ValueError('Please select a valid normalization')
    return img_patch

def enhance_flair(lesion, normalization=None):
    if normalization is not None:
        return normalize_patch(cv2.blur(lesion[FLAIR] ** 2, (6,6)), normalization)
    return cv2.blur(lesion[FLAIR] ** 2, (6,6))

def get_folds_structure(version = "all"):
    '''
        Function that gets the structure of all the folds.
    '''
    
    p0_chuv = list((DATASET_CHUV_ID, pat) for pat in ('011', '012', '016', '017', '021', '024', '028', '030', '032', '039', '041', '050', '053'))
    p1_chuv = list((DATASET_CHUV_ID, pat) for pat in ('001', '008', '015', '020', '022', '025', '031', '034', '035', '037', '040', '042', '043', '046'))
    p2_chuv = list((DATASET_CHUV_ID, pat) for pat in ('003', '005', '010', '013', '014', '023', '027', '029', '044', '045', '047', '048', '052', '054', '055'))
    p3_chuv = list((DATASET_CHUV_ID, pat) for pat in ('002', '004', '006', '007', '009', '018', '019', '026', '033', '036', '038', '049', '051'))
    
    p0_basel = list((DATASET_BASEL_ID, pat) for pat in ('063', '064', '065', '066', '067', '069', '075', '078', '080', '097', '099', '100', '108', '112', '114', '116', '120', '125', '131'))
    p1_basel = list((DATASET_BASEL_ID, pat) for pat in ('057', '059', '060', '062', '072', '073', '081', '082', '092', '095', '098', '107', '110', '122', '124', '126', '129', '130', '132'))
    p2_basel = list((DATASET_BASEL_ID, pat) for pat in ('068', '077', '079', '083', '084', '093', '096', '103', '109', '113', '115', '118'))
    p3_basel = list((DATASET_BASEL_ID, pat) for pat in ('056', '058', '061', '070', '071', '074', '076', '085', '094', '101', '102', '104', '105', '106', '117', '121', '123', '127', '128'))
    
    auto_split_patients = list((DATASET_BASEL_ID, pat) for pat in ('056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '104', '105', '106', '108', '109', '110', '112'))
    
    
    if "all_auto" in version:
        return [list([el for el in fold if el in auto_split_patients]) for fold in get_folds_structure(version = "all")]
    
    elif "all" in version:
        return [p0_chuv + p0_basel, 
                p1_chuv + p1_basel, 
                p2_chuv + p2_basel, 
                p3_chuv + p3_basel]
    
    elif version == "basel":
        return [p0_basel, p1_basel, p2_basel, p3_basel]
    elif version == "chuv":
        return [p0_chuv, p1_chuv, p2_chuv, p3_chuv]
    
    elif "pilot_training" in version:
        return [list((DATASET_BASEL_ID, pat) for pat in ('056', '057', '058', '059', '060', '061', '062', '063', '064', '072', '073', '075', '076', '077', '078', '079', '080', '081', '082', '092', '094', '095', '096', '097', '098', '099', '100', '101', '102', '104', '105')), # training
               list((DATASET_BASEL_ID, pat) for pat in ('071', '083', '106', '108', '109', '112', '110'))] # validation
    elif "pilot_testing" in version:
        return [list((DATASET_BASEL_ID, pat) for pat in ('065', '066', '067', '068', '069', '070', '074', '093')), ] # testing
    
    elif "nih7T_testing" in version:
        nih = list((DATASET_NIH7T_ID, pat) for pat in ('001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020'))
        return [nih[0:5], nih[5:10], nih[10:15], nih[15:]]# testing
    
    # To TEST on the training samples
    elif version == "basel_training":
        return [p1_basel + p2_basel + p3_basel, p0_basel + p2_basel + p3_basel, p0_basel + p1_basel + p3_basel, p0_basel + p1_basel + p2_basel]
    elif version == "chuv_training":
        return [p1_chuv + p2_chuv + p3_chuv, p0_chuv + p2_chuv + p3_chuv, p0_chuv + p1_chuv + p3_chuv, p0_chuv + p1_chuv + p2_chuv]
    
    # Only those with MPRAGE available. Protocol 1 and Protocol 2.
    elif version == "mprage_p1":
        return [list((DATASET_CHUV_ID, pat) for pat in ('012', '016', '021', '028', '030', '032', '039')), 
                list((DATASET_CHUV_ID, pat) for pat in ('001', '008', '020', '034', '037', '040', '042', '046')), 
                list((DATASET_CHUV_ID, pat) for pat in ('003', '010', '014', '027', '029', '044', '045', '047', '048')), 
                list((DATASET_CHUV_ID, pat) for pat in ('004', '006', '007', '009', '018', '019', '026', '036', '038'))]
    elif version == "mprage_p2":
        return [list((DATASET_CHUV_ID, pat) for pat in ('011', '017', '024', '041', '050', '053')), 
                list((DATASET_CHUV_ID, pat) for pat in ('015', '022', '025', '031', '035', '043')), 
                list((DATASET_CHUV_ID, pat) for pat in ('005', '013', '023', '052', '054', '055')), 
                list((DATASET_CHUV_ID, pat) for pat in ('002', '033', '049', '051'))]
    
    raise Exception("Non supported fold split.")




def extract_patch_from_bigger_lesion(lesion, mov, goal_patch_size, contrasts, normalization):
    '''
        Function that receives a lesion from bigger patch (34x34x34) for example and crops it to "goal_patch_size".
        The crop can be centered (mov = (0,0,0)) or moved, specifying in "mov" the translated new center.
        Returns the contrasts ready to put as input in the network.
    '''
    goal_patch_size = np.array(goal_patch_size)
    mov = np.array(mov)
    
    min_ = (lesion[0,:].shape - goal_patch_size) // 2 + mov
    max_ = min_ + goal_patch_size
    
    assert (min_ >= (0,0,0)).all() and (max_ <= lesion[0,:].shape).all()
        
    lesion = lesion[:, min_[0]:max_[0], min_[1]:max_[1], min_[2]:max_[2]]
    
    return process_patch(lesion, contrasts, normalization, goal_patch_size)


def enhance_flair(image, normalization, blur=False):
    aux = image.copy()
    for i in range(3):
        aux *= image
    if blur:
        ker = (4, 4)
        aux = ndimage.rotate(cv2.blur(ndimage.rotate(aux, 90.0, axes=(0,2), reshape=False, order=0, mode='nearest'), ker), -90.0, axes=(0,2), reshape=False, order=0, mode='nearest')
    return normalize_patch(aux, normalization)


def enhance_flairstar(image, normalization):
    image = image.copy()
    aux = image*image
    ker = (6, 6)
    aux = cv2.blur(aux, ker)
    aux = ndimage.rotate(cv2.blur(ndimage.rotate(aux, 90.0, axes=(0,2), reshape=False, order=0, mode='nearest'), ker), -90.0, axes=(0,2), reshape=False, order=0, mode='nearest')
    return normalize_patch(aux, normalization)


#___________________ BIDS STUFF _____________________

def validate_BIDS(details=False):
    for layout in AVAILABLE_DATASETS:
        print(layout)
        ok = True
        subjects = layout.get_subjects()
        for contrast in CONTRASTS.keys():
            subjects_ok = layout.get(return_type="id", target="subject", **CONTRASTS[contrast])
            missing = tuple(np.array(sorted(set(subjects) - set(subjects_ok))).astype(int))
            if len(missing) > 0:
                ok = False
                if details:
                    print(f"{contrast} - Missing {len(missing)}/{len(subjects)}: {missing}")
                else:
                    print(f"{contrast} - Missing {len(missing)}/{len(subjects)}")
        print(f"[STATUS] {'OK' if ok else 'incompleted'}\n")
        

def generate_BIDS_path(db_id, *, subject=None, scope=None, suffix=None, acquisition=None, extension=None):
    derivative = scope is not None and scope != "raw"
    if subject is None or scope is None or suffix is None or extension is None:
        return False
    
    path = f"{'derivatives/' + scope + '/' if derivative else ''}sub-{subject}/ses-01/{'anat/' if not derivative else ''}sub-{subject}_ses-01{'_acq-' + acquisition if acquisition is not None else ''}_{suffix}.{extension}"
    return os.path.join(AVAILABLE_DATASETS_ROOTS[db_id], path)


def compute_auc_and_threshold(truth, preds, accepted_fpr=0.05):
    fpr, tpr, thresholds = metrics.roc_curve(truth, preds)
    auc = metrics.auc(fpr, tpr)
    index = np.abs(fpr - accepted_fpr).argmin()
    th = thresholds[index]
    return auc, th




# _______________ DEFORMATION STUFF __________________

def generate_deformed_lesions_files(random_seed, from_size, goal_size):
    from_size, goal_size = np.array(from_size), np.array(goal_size)
    #metadata = read_patients_metadata(goal_size)
    
    print(f"[INFO] Generating deformed lesions with SEED KERNEL: {random_seed}")
    pipe = DERIVATIVES["LESIONS_DEF"]["pipeline"].format(goal_size[0], goal_size[1], goal_size[2], random_seed)
    pipe_from = DERIVATIVES["LESIONS"]["pipeline"].format(from_size[0], from_size[1], from_size[2])
    #print(pipe)
    for db_id in range(len(AVAILABLE_DATASETS)):
        dataset = AVAILABLE_DATASETS[db_id]
        for pat in tqdm(sorted(dataset.get_subjects())):
            pat = int(pat)
            mask_paths = dataset.get(return_type="filename", subject=f"{pat:03d}", **CONTRASTS["MASK"])
            if len(mask_paths) != 1:
                print(f"[ERROR] Check mask of patient {pat}.")
                continue
            mask_path = mask_paths[0]
            path, extension = generate_and_get_folder_for_lesions(mask_path, goal_size, int(pat), deformed=random_seed)
            #print(path)
            
            # We only keep the contrasts missing
            to_run = list(PURE_CONTRASTS)
            def_lesions = {} # where we will store lesions to save
            for c in PURE_CONTRASTS:
                acq, suff = CONTRASTS[c]["acquisition"], CONTRASTS[c]["suffix"]
                already_generated = len(dataset.get(return_type="filename", subject=f"{pat:03d}", scope=pipe, acquisition=acq, suffix=suff, extension=DERIVATIVES["LESIONS"]["extension"])) != 0
                contrast_available = len(dataset.get(return_type="filename", subject=f"{pat:03d}", scope=pipe_from, acquisition=acq, suffix=suff, extension=DERIVATIVES["LESIONS"]["extension"])) != 0
                if already_generated or not contrast_available:
                    to_run.remove(c)
                    #print(f"To skip: {c}")
                else:
                    def_lesions[c] = {}
                    
            if len(to_run) == 0:
                print(f"[INFO] Patient {pat} skipped.")
                continue
            
            result = load_patient_lesions(db_id, pat, from_size, None, only_cleaned=True)
            if type(result) is int:
                print(f"[WARNING] Patient {pat} skipped. Lesions file could not be read.")
                continue
                
            _, _, pat_lesions, metadata = result
            # We deform all lesions for the missing contrasts
            for les_id in pat_lesions.keys():
                # Lesion seed: int(AXYYYZZZZ) where A kernel, X dataset, YYY patient, ZZZZ lesion
                lesion_seed = int(f"{random_seed}{db_id}{pat:03d}{les_id}")
                np.random.seed(lesion_seed)
                
                les = pat_lesions[les_id]
                def_lesion = deform_and_crop_lesion(les_id, les, goal_size)
                
                if def_lesion is not None:
                    for c in to_run:
                        def_lesions[c][les_id] = def_lesion[c]
            
            # SAVE
            for c in to_run:
                filename = os.path.join(path, get_filename(pat, acquisition = CONTRASTS[c]["acquisition"], suffix=CONTRASTS[c]["suffix"], extension=extension))
                #print(filename)
                __save_patches(filename, def_lesions[c])
