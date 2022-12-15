import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
import sys
sys.path.append("..")
from testing import run_test

# ARCHS
import archs.rimnet as rimnet
import archs.monomodal as monomodal
import archs.bimodal as monomodal_attention
import archs.rimnet_bi as binet
import archs.bimodal as bimodal


NETWORKS_TO_TEST = [
    #{"network_name": "auto_binet_phase_flair", "network": binet, "folds_version": "pilot_testing", "contrasts": ["T2STAR_PHASE", "FLAIR"] },
    #{"network_name": "auto_mono_phase", "network": monomodal, "folds_version": "pilot_testing", "contrasts": ["T2STAR_PHASE", ] },
    {"network_name": "mono_mp2uni_basel", "network": monomodal, "folds_version": "nih7T_testing", "contrasts": ["MP2RAGE_UNI_ORIGINAL", ], "ensemble": True  },
    {"network_name": "mono_mp2map_basel", "network": monomodal, "folds_version": "nih7T_testing", "contrasts": ["MP2RAGE_T1MAP_ORIGINAL", ], "ensemble": True  },
]
SPLIT_VERSION = "annotated_7T_rimpos"


for config in NETWORKS_TO_TEST:
    config["normalization_type"] = "local_max"

    
results = run_test(NETWORKS_TO_TEST, from_segmentation=True, split_version=SPLIT_VERSION)