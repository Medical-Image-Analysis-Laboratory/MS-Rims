import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
import sys
sys.path.append("../../")
from testing import run_test

# ARCHS
import archs.rimnet as rimnet
import archs.monomodal as monomodal
import archs.monomodal_attention as monoatt
import archs.rimnet_bi as binet
import archs.bimodal as bimodal

NETWORKS_TO_TEST = [

    #{"network_name": "rimnet_flair", "network": rimnet, "folds_version": "all", "contrasts": ["T2STAR_PHASE", "T2STAR_MAG", "FLAIR"]},
    #{"network_name": "rimnet_eflair", "network": rimnet, "folds_version": "all", "contrasts": ["T2STAR_PHASE", "T2STAR_MAG", "eFLAIR"]},
    #{"network_name": "rimnet_eflair_bl", "network": rimnet, "folds_version": "all", "contrasts": ["T2STAR_PHASE", "T2STAR_MAG", "eFLAIR_bl"]},
    #{"network_name": "rimnet_flairstar", "network": rimnet, "folds_version": "all", "contrasts": ["T2STAR_PHASE", "T2STAR_MAG", "FLAIRSTAR"]},
    
    
    #{"network_name": "rimnet_flair_basel", "network": rimnet, "folds_version": "chuv", "contrasts": ["T2STAR_PHASE", "T2STAR_MAG", "FLAIR"], "ensemble": True},
    #{"network_name": "mono_flair_basel", "network": monomodal, "folds_version": "chuv", "contrasts": ["FLAIR", ], "ensemble": False },
    #{"network_name": "mono_phase_basel", "network": monomodal, "folds_version": "chuv", "contrasts": ["T2STAR_PHASE", ], "ensemble": False },
    #{"network_name": "mono_t2star_basel", "network": monomodal, "folds_version": "chuv", "contrasts": ["T2STAR_MAG", ], "ensemble": False },

    #{"network_name": "binet_t2star_flair_basel", "network": binet, "folds_version": "chuv", "contrasts": ["T2STAR_MAG", "FLAIR"], "ensemble": False },
    #{"network_name": "binet_phase_flair_basel", "network": binet, "folds_version": "chuv", "contrasts": ["T2STAR_PHASE", "FLAIR"], "ensemble": False },
    #{"network_name": "binet_phase_t2star_basel", "network": binet, "folds_version": "chuv", "contrasts": ["T2STAR_PHASE", "T2STAR_MAG"], "ensemble": False },

    # RIMNET 2-MODALITIES EVALUATION
    #{"network_name": "binet_phase_flair", "network": binet, "folds_version": "all", "contrasts": ["T2STAR_PHASE", "FLAIR"]},
    #{"network_name": "binet_t2star_flair", "network": binet, "folds_version": "all", "contrasts": ["T2STAR_MAG", "FLAIR"]},
    #{"network_name": "binet_phase_t2star", "network": binet, "folds_version": "all", "contrasts": ["T2STAR_PHASE", "T2STAR_MAG"]},

    {"network_name": "bimodal_phase_flair", "network": bimodal, "folds_version": "all", "contrasts": ["T2STAR_PHASE", "FLAIR"] },
    #{"network_name": "bimodal_t2star_flair", "network": bimodal, "folds_version": "all", "contrasts": ["T2STAR_MAG", "FLAIR"] },
    #{"network_name": "bimodal_phase_t2star", "network": bimodal, "folds_version": "all", "contrasts": ["T2STAR_PHASE", "T2STAR_MAG"] },

    #{"network_name": "monoatt_phase_flair", "network": monoatt, "folds_version": "all", "contrasts": ["T2STAR_PHASE", "FLAIR"] },
    #{"network_name": "monoatt_t2star_flair", "network": monoatt, "folds_version": "all", "contrasts": ["T2STAR_MAG", "FLAIR"] },
    #{"network_name": "monoatt_phase_t2star", "network": monoatt, "folds_version": "all", "contrasts": ["T2STAR_PHASE", "T2STAR_MAG"] },

    #{"network_name": "monomodal_flair", "network": monomodal, "folds_version": "all", "contrasts": ["FLAIR", ] },
    #{"network_name": "monomodal_flairstar", "network": monomodal, "folds_version": "all", "contrasts": ["FLAIRSTAR", ] },
    #{"network_name": "monomodal_phase", "network": monomodal, "folds_version": "all", "contrasts": ["T2STAR_PHASE", ] },
    #{"network_name": "monomodal_t2star", "network": monomodal, "folds_version": "all", "contrasts": ["T2STAR_MAG", ] },
    
    # MP2RAGE EVALUATION
    #{"network_name": "mono_flair_basel", "network": monomodal, "folds_version": "basel", "contrasts": ["FLAIR", ] },
    #{"network_name": "mono_mp2uni_basel", "network": monomodal, "folds_version": "basel", "contrasts": ["MP2RAGE_UNI", ] },
    #{"network_name": "mono_mp2t1map_basel", "network": monomodal, "folds_version": "basel", "contrasts": ["MP2RAGE_T1MAP", ] },
    #{"network_name": "mono_phase_basel", "network": monomodal, "folds_version": "basel", "contrasts": ["T2STAR_PHASE", ], "ensemble": True  },
    #{"network_name": "mono_t2star_basel", "network": monomodal, "folds_version": "basel", "contrasts": ["T2STAR_MAG", ] }, 

    #{"network_name": "binet_t2star_flair_basel", "network": binet, "folds_version": "basel", "contrasts": ["T2STAR_MAG", "FLAIR"] },
    #{"network_name": "binet_phase_flair_basel", "network": binet, "folds_version": "basel", "contrasts": ["T2STAR_PHASE", "FLAIR"] },
    #{"network_name": "binet_phase_mp2uni_basel", "network": binet, "folds_version": "basel", "contrasts": ["T2STAR_PHASE", "MP2RAGE_UNI"] },
    #{"network_name": "binet_phase_t2star_basel", "network": binet, "folds_version": "basel", "contrasts": ["T2STAR_PHASE", "T2STAR_MAG"] },

    # CROSS CENTER EVALUATION
    #{"network_name": "mono_flair_basel", "network": monomodal, "folds_version": "chuv", "contrasts": ["FLAIR", ], "ensemble": True },
    #{"network_name": "mono_phase_basel", "network": monomodal, "folds_version": "chuv", "contrasts": ["T2STAR_PHASE", ], "ensemble": True },
    #{"network_name": "mono_t2star_basel", "network": monomodal, "folds_version": "chuv", "contrasts": ["T2STAR_MAG", ], "ensemble": True },

    #{"network_name": "binet_t2star_flair_basel", "network": binet, "folds_version": "chuv", "contrasts": ["T2STAR_MAG", "FLAIR"], "ensemble": True },
    #{"network_name": "binet_phase_flair_basel", "network": binet, "folds_version": "chuv", "contrasts": ["T2STAR_PHASE", "FLAIR"], "ensemble": True },
    #{"network_name": "binet_phase_t2star_basel", "network": binet, "folds_version": "chuv", "contrasts": ["T2STAR_PHASE", "T2STAR_MAG"], "ensemble": True },
]


for config in NETWORKS_TO_TEST:
    config["normalization_type"] = "local_max"

run_test(NETWORKS_TO_TEST)
    