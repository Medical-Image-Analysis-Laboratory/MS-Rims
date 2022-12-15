import os
from bids import BIDSLayout
import numpy as np



print("Loading configuration...")

BIDS_VERSION = "1.2.2"
WORKING_MODE_SKULL_STRIPPED = False

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# CHECK AT FIRST USE
#PATH_DATA= os.path.join(ROOT_DIR, '/media/german/Germs_DATA/01_dataset/')
PATH_DATA= '/data/'
MODELS_LOAD_FROM = "/models/"
MODELS_FIGS_SAVE_TO = "/media/german/Germs_DATA/05_figures/"

# This is for TRAINING, so only for the cluster folders.
PATH_SUMMARY= os.path.join(PATH_DATA, 'tensorboard/')
PATH_MODEL= os.path.join(ROOT_DIR, 'model/')
#PATH_CHECKPOINTS= os.path.join(PATH_DATA, 'checkpoints/')
PATH_CHECKPOINTS= MODELS_LOAD_FROM
PATH_TRANSLATION= os.path.join(PATH_DATA, 'translation_analysis/')
PATH_TEST= os.path.join(ROOT_DIR, 'figs/')
PATH_TEST_PREDS = os.path.join(PATH_DATA, "test/")

# DATASETS INCLUDED
#DATASET_BASEL_ID = 0 # Basel 3T dataset
#DATASET_BASEL_ROOT = os.path.join(PATH_DATA, "BASEL_INSIDER")
#DATASET_BASEL = BIDSLayout(DATASET_BASEL_ROOT, derivatives=True)
#DATASET_CHUV_ID = 1 # Lausanne (CHUV) 3T dataset
#DATASET_CHUV_ROOT = os.path.join(PATH_DATA, "CHUV_RIM")
#DATASET_CHUV = BIDSLayout(DATASET_CHUV_ROOT, derivatives=True)
#DATASET_NIH7T_ID = 2 # NIH 7T dataset
#DATASET_NIH7T_ROOT = os.path.join(PATH_DATA, "NIH_7T")
#DATASET_NIH7T = BIDSLayout(DATASET_NIH7T_ROOT, derivatives=True)
# available datasets
#AVAILABLE_DATASETS = [DATASET_BASEL, DATASET_CHUV, DATASET_NIH7T]
#AVAILABLE_DATASETS = [DATASET_BASEL]
#AVAILABLE_DATASETS_ROOTS = [DATASET_BASEL_ROOT, DATASET_CHUV_ROOT, DATASET_NIH7T_ROOT]


PATCH_SIZE_DEFORMATIONS = np.array((40, 40, 40))
PATCH_SIZE = np.array((34, 34, 34))
PATCH_SIZE_TRAINING = np.array((28, 28, 28))

LABEL_NON_RIM = (1, 0)
LABEL_RIM = (0, 1)
N_CLASSES = 2

# EXCLUSION CRITERIA
EXCLUSION_OTHERS = -1
EXCLUSION_SMALL = 0
EXCLUSION_BIG = 1
EXCLUSION_RIM_INTRUSION = 2
EXCLUSION_AIR_ARTIFACT = 3

def get_exclusion_reason(reason):
    if reason == EXCLUSION_OTHERS:
        return "Others"
    elif reason == EXCLUSION_SMALL:
        return "< 45"
    elif reason == EXCLUSION_BIG:
        return "> 10000"
    elif reason == EXCLUSION_RIM_INTRUSION:
        return "Rim intrusion"
    elif reason == EXCLUSION_AIR_ARTIFACT:
        return "Air artifact"

# CONTRAST INDEX IN FILENAME_LESIONS
PHASE = 0
MASK = 1
T2 = 2
FLAIR = 3
QSM = 4
FLAIR_MASKED = -1 # SPECIAL PREPROCESSING

def get_filename(pat, *, ses=1, acquisition=None, suffix=None, extension="nii.gz"):
    if acquisition is None:
        return f"sub-{pat:03d}_ses-{ses:02d}_{suffix}.{extension}"
    return f"sub-{pat:03d}_ses-{ses:02d}_acq-{acquisition}_{suffix}.{extension}"

import archs.rimnet as rimnet
import archs.monomodal as monomodal
import archs.monomodal_attention as monoatt
import archs.rimnet_bi as binet
import archs.bimodal as bimodal

# Contrasts taken into account in the BIDS system
NETWORKS = {'auto_binet_phase_flair_morpho_v01_DAv6':(binet,  ["T2STAR_PHASE", "FLAIR"]), 
            'auto_binet_phase_flair_morpho_v02_DAv6':(binet,  ["T2STAR_PHASE", "FLAIR"]),
            'auto_binet_phase_flair_morpho_v02_maxDA_DAv6':(binet,  ["T2STAR_PHASE", "FLAIR"]),
            'auto_binet_phase_flair_morpho_v02_maxDA_DAv6_DEF5':(binet,  ["T2STAR_PHASE", "FLAIR"]),
            'auto_binetplus_phase_flair_morpho_v02_maxDA_DAv6_DEF5':(binet,  ["T2STAR_PHASE", "FLAIR"]),
            'bimodal_phase_flair':(bimodal,  ["T2STAR_PHASE", "FLAIR"]),
            'bimodal_phase_t2star':(bimodal,["T2STAR_PHASE", "T2STAR_MAG"]),
            'bimodal_t2star_flair':(bimodal, ["T2STAR_MAG", "FLAIR"]),
            'binet_phase_flair':(binet, ["T2STAR_PHASE", "FLAIR"]),
            'binet_phase_flair_basel':(binet,  ["T2STAR_PHASE", "FLAIR"]),
            'binet_phase_flair_chuv':(binet,  ["T2STAR_PHASE", "FLAIR"]),
            'binet_phase_mp2map_basel':(binet,["T2STAR_PHASE", "MP2RAGE_T1MAP"]), 
            'binet_phase_mp2uni_basel':(binet, ["T2STAR_PHASE", "MP2RAGE_UNI"]),
            'binet_phase_t2star':(binet,["T2STAR_PHASE", "T2STAR_MAG"]),
            'binet_phase_t2star_basel':(binet,["T2STAR_PHASE", "T2STAR_MAG"]),
            'binet_t2star_flair':(binet, ["T2STAR_MAG", "FLAIR"]),
            'binet_t2star_flair_basel':(binet, ["T2STAR_MAG", "FLAIR"]),
            'mono_flair_basel':(monomodal, ["FLAIR", ]) ,
            'mono_mp2map_basel':(monomodal,["MP2RAGE_T1MAP", ]),
            'mono_mp2map_basel_interp':(monomodal, ["MP2RAGE_T1MAP", ]),
            'mono_mp2uni_basel':(monomodal,  ["MP2RAGE_UNI", ]),
            'mono_phase_basel':(monomodal, ["T2STAR_PHASE", ]) ,
            'mono_t2star_basel':(monomodal,  ["T2STAR_MAG", ]),
            'mono_t2star_basel_2':(monomodal, ["T2STAR_MAG", ]),
            'monoatt_phase_flair':(monoatt, ["T2STAR_PHASE", "FLAIR"]) ,
            'monoatt_phase_t2star':(monoatt,["T2STAR_PHASE", "T2STAR_MAG"]),
            'monoatt_t2star_flair':(monoatt, ["T2STAR_MAG", "FLAIR"]),
            'monomodal_flair':(monomodal, ["FLAIR", ] ),
            'monomodal_flairstar':(monomodal, ["FLAIRSTAR", ]) ,
            'monomodal_phase':(monomodal, ["T2STAR_PHASE", ]) ,
            'monomodal_t2star':(monomodal, ["T2STAR_MAG", ]),
            'rimnet_eflair':(rimnet,  ["T2STAR_PHASE", "T2STAR_MAG", "eFLAIR"]) ,
            'rimnet_eflair_bl':(rimnet,  ["T2STAR_PHASE", "T2STAR_MAG", "eFLAIR_bl"]) ,
            'rimnet_flair':(rimnet,["T2STAR_PHASE", "T2STAR_MAG", "FLAIR"]),
            'rimnet_flair_basel':(rimnet,["T2STAR_PHASE", "T2STAR_MAG", "FLAIR"]),
            'rimnet_flair_chuv':(rimnet,["T2STAR_PHASE", "T2STAR_MAG", "FLAIR"]) ,
            'rimnet_flairstar':(rimnet,["T2STAR_PHASE", "T2STAR_MAG", "FLAIRSTAR"]) ,
            'rimnet_ph_mp2u_fl_basel':(rimnet,["T2STAR_PHASE", "MP2RAGE_UNI", "FLAIR"]),
            'rimnet_v2_ph_mp2u_fl_basel':(rimnet,["T2STAR_PHASE", "MP2RAGE_UNI", "FLAIR"]),
            'rimnet_v3_ph_mp2u_fl_basel':(rimnet,["T2STAR_PHASE", "MP2RAGE_UNI", "FLAIR"]),
            'sk_binet_phase_mp2uni_basel':(binet, ["T2STAR_PHASE", "MP2RAGE_UNI"]),
            'sk_mono_mp2uni':(monomodal, ["MP2RAGE_UNI",]),
            'sk_mono_mp2uni_basel':(monomodal, ["MP2RAGE_UNI",]),
            'sk_mono_phase_basel':(monomodal, ["T2STAR_PHASE", ])}

CONTRASTS = {
    # >>>> MASK + SEGMENTATION from Francesco's CNN
    "MASK": { "acquisition": None, "suffix": "mask", "scope": "rims_annotations", "extension": "nii.gz" },
    "SEGMENTATION": { "acquisition": None, "suffix": "segmentation", "scope": "segmentations", "extension": "nii.gz" },
    
    # >>>> Contrasts available
    "T2STAR_PHASE": { "acquisition": "phase", "suffix": "T2star", "scope": "raw", "extension": "nii.gz"},
    "T2STAR_MAG": { "acquisition": "mag", "suffix": "T2star", "scope": "raw", "extension": "nii.gz"},
    "FLAIRSTAR": { "acquisition": "star", "suffix": "FLAIR", "scope": "raw", "extension": "nii.gz"},
    "FLAIR": { "acquisition": None, "suffix": "FLAIR", "scope": "registrations_to_T2star", "extension": "nii.gz"},
    #"FLAIR_ORIGINAL": { "acquisition": None, "suffix": "FLAIR", "scope": "raw", "extension": "nii.gz"},
    "MPRAGE": { "acquisition": "MPRAGE", "suffix": "T1map", "scope": "registrations_to_T2star", "extension": "nii.gz"},
    #"MPRAGE_ORIGINAL": { "acquisition": "MPRAGE", "suffix": "T1map", "scope": "raw", "extension": "nii.gz"},
    "MP2RAGE_UNI": { "acquisition": "MP2RAGEuni", "suffix": "T1map", "scope": "registrations_to_T2star", "extension": "nii.gz"},
    "MP2RAGE_T1MAP": { "acquisition": "MP2RAGE", "suffix": "T1map", "scope": "registrations_to_T2star", "extension": "nii.gz"},
    "MP2RAGE_UNI_ORIGINAL": { "acquisition": "MP2RAGEuni", "suffix": "T1map", "scope": "raw", "extension": "nii.gz"},
    "MP2RAGE_T1MAP_ORIGINAL": { "acquisition": "MP2RAGE", "suffix": "T1map", "scope": "raw", "extension": "nii.gz"},
    
    # 3T -> 7T NIH
    "3T_FLAIR": { "acquisition": None, "suffix": "FLAIR", "scope": "3T_data", "extension": "nii.gz"},
    "3T_MP2RAGE_UNI": { "acquisition": "MP2RAGEuni", "suffix": "T1map", "scope": "3T_data", "extension": "nii.gz"},
    "3T_SEGMENTATION": { "acquisition": "MP2RAGEuni", "suffix": "T1map", "scope": "3T_data", "extension": "nii.gz"},
    "7TREG_3T_MP2RAGE_UNI": { "acquisition": "MP2RAGEuni", "suffix": "T1map", "scope": "7T_from_3T_data", "extension": "nii.gz"},
    "MP2RAGE_UNI_ORIGINAL": { "acquisition": "MP2RAGEuni", "suffix": "T1map", "scope": "raw", "extension": "nii.gz"},
    "MP2RAGE_T1MAP_ORIGINAL": { "acquisition": "MP2RAGE", "suffix": "T1map", "scope": "raw", "extension": "nii.gz"},
    #"MP2RAGE_UNI_ORIGINAL_SK": { "acquisition": "MP2RAGEuni", "suffix": "T1mapSK", "scope": "raw", "extension": "nii.gz"},
    
    #"QSM": { "acquisition": "QSM", "suffix": "T2star", "scope": "raw", "extension": "nii.gz"},
    
    
    # >>>> MNI extension
    #"FLAIR_MNI": { "acquisition": None, "suffix": "FLAIR", "scope": "MNI_space", "extension": "nii.gz"},
    #"MASK_MNI": { "acquisition": None, "suffix": "mask", "scope": "MNI_space", "extension": "nii.gz" },
    #"MASK_FREESURFER_ORIGINAL": { "acquisition": None, "suffix": "segmentation", "scope": "freesurfer_segmentation", "extension": "nii.gz" },
    #"MASK_FREESURFER": { "acquisition": None, "suffix": "segmentationREG", "scope": "freesurfer_segmentation", "extension": "nii.gz" },
    
    
    # >>>> Skull stripped extension for MP2RAGE SYNTHETIC
    #"MASK_SK": { "acquisition": None, "suffix": "mask", "scope": "rims_annotations", "extension": "nii.gz" },
    #"MASK_BRAIN": { "acquisition": None, "suffix": "brainmaskREG", "scope": "synthetic_mp2rage", "extension": "nii.gz" },
    #"T2STAR_PHASE_SK": { "acquisition": "phase", "suffix": "T2star", "scope": "skull_stripped", "extension": "nii.gz"},
    #"T2STAR_MAG_SK": { "acquisition": "mag", "suffix": "T2star", "scope": "skull_stripped", "extension": "nii.gz"},
    #"FLAIR_SK": { "acquisition": None, "suffix": "FLAIR", "scope": "skull_stripped", "extension": "nii.gz"},
    #"MPRAGE_SK": { "acquisition": "MPRAGE", "suffix": "T1map", "scope": "skull_stripped", "extension": "nii.gz" },
    #"MP2RAGE_UNI_SK": { "acquisition": "MP2RAGEuni", "suffix": "T1map", "scope": "skull_stripped", "extension": "nii.gz"},
    #"MP2RAGE_SYNTHETIC": { "acquisition": "MP2RAGEsynthetic", "suffix": "T1map", "scope": "skull_stripped", "extension": "nii.gz" },
    
    
    # >>>> Fully automatic pipeline extension
    #"PMAP_ORIGINAL": { "acquisition": None, "suffix": "probabilitiesmap", "scope": "segmentation_probability_maps", "extension": "nii.gz" },
    "PMAP": { "acquisition": None, "suffix": "probabilitiesmapREG", "scope": "segmentation_probability_maps", "extension": "nii.gz" },
    "EXPERTS_ANNOTATIONS": { "acquisition": None, "suffix": "expertsannotations", "scope": "expert_annotations", "extension": "nii.gz" },
}

SPLIT_LESIONS_METADATA = {
    "pmaps_v02": { "folder_name": "autosplit_lesions_v02", "pipeline": "autosplit_lesions_v02", "suffix": "splitmask"},
    "pmaps_v03": { "folder_name": "autosplit_lesions_v03", "pipeline": "autosplit_lesions_v03", "suffix": "splitmask"},
    "pmaps_v04": { "folder_name": "autosplit_lesions_v04", "pipeline": "autosplit_lesions_v04", "suffix": "splitmask"},
    "pmaps_v05": { "folder_name": "autosplit_lesions_v05", "pipeline": "autosplit_lesions_v05", "suffix": "splitmask"},
    "annotated_7T_rimpos": { "folder_name": "autosplit_lesions_7T_rimpos", "pipeline": "autosplit_lesions_7T_rimpos", "suffix": "splitmask"},
    "morpho_v01": { "folder_name": "autosplit_lesions_morpho_v01", "pipeline": "autosplit_lesions_morpho_v01", "suffix": "splitmask"},
    "morpho_v02": { "folder_name": "autosplit_lesions_morpho_v02", "pipeline": "autosplit_lesions_morpho_v02", "suffix": "splitmask"},
    "morpho_v02_maxDA": { "folder_name": "autosplit_lesions_morpho_v02_maxDA", "pipeline": "autosplit_lesions_morpho_v02_maxDA", "suffix": "splitmask"},
}

PATIENTS_METADATA = {"folder_name": "lesions_{}_{}_{}", 
                     "pipeline": "lesions_extractor_{}_{}_{}", 
                     "extension": "json"}

# The ones loaded with load_lesions

if not WORKING_MODE_SKULL_STRIPPED:
    PURE_CONTRASTS = ["T2STAR_PHASE", "T2STAR_MAG", "FLAIRSTAR", "FLAIR", "MP2RAGE_UNI", "MP2RAGE_T1MAP", "MASK", "MP2RAGE_UNI_ORIGINAL", "MP2RAGE_T1MAP_ORIGINAL", "7TREG_3T_MP2RAGE_UNI"]
    # The ones available to use as training (pure + generated)
    IMPLEMENTED_CONTRASTS = PURE_CONTRASTS + ["eFLAIRSTAR", "eFLAIR", "eFLAIR_bl"]
    DERIVATIVES = {
        "LESIONS": {"folder_name": "lesions_{}_{}_{}", "pipeline": "lesions_extractor_{}_{}_{}", "extension": "dat"},
        "LESIONS_DEF": {"folder_name": "lesions_{}_{}_{}_DEF-{}", "pipeline": "lesions_extractor_{}_{}_{}_DEF-{}", "extension": "dat"},
    }
else:
    PURE_CONTRASTS = ["T2STAR_PHASE_SK", "T2STAR_MAG_SK", "FLAIR_SK", "MPRAGE_SK", "MP2RAGE_UNI_SK", "MP2RAGE_SYNTHETIC", "MASK_SK"]
    # The ones available to use as training (pure + generated)
    IMPLEMENTED_CONTRASTS = PURE_CONTRASTS
    DERIVATIVES = {
        "LESIONS": {"folder_name": "lesions_{}_{}_{}", "pipeline": "lesions_extractor_{}_{}_{}", "extension": "sk.dat"},
        "LESIONS_DEF": {"folder_name": "lesions_{}_{}_{}_DEF-{}", "pipeline": "lesions_extractor_{}_{}_{}_DEF-{}", "extension": "sk.dat"},
    }



print("Configuration loaded successfully!\n_____________________________\n")

