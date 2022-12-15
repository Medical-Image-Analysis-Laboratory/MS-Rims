
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("..")
from config import *
import logging
import time
from datetime import datetime
from training import train_fold_ncv, get_folded_data, print_all
from utils import load_lesions

parser = argparse.ArgumentParser(description="Train model.")
parser.add_argument('fold', help="Num of fold to train.", type=int)
parser.add_argument('gpu', nargs='?', help="GPU device to use.", type=int, default=0)
args = parser.parse_args()

if args.gpu not in (0, 1):
    raise(Exception("GPU value is not valid"))
if args.fold > 4 or args.fold < 0:
    raise(Exception("Fold not valid"))

FOLD = args.fold
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)


NETWORK_CONTRASTS = ["MP2RAGE_T1MAP", ]#["T2STAR_PHASE", "T2STAR_MAG", "FLAIR"]
LESIONS = [None, 1, 2, 3] # non and deformed versions 1, 2, 3
NETWORK_NAME = "mono_mp2map_basel"
#NETWORK_NAME = "rimnet_t2star_flair"
import archs.monomodal as NETWORK

# To avoid inner cross-validation. NOT RECOMMENDED
EPOCHS = None #[12, 12, 14, 14, 14]

LOAD_ASYNCR = True
DA_ONLINE_STRATEGY = 'v4'
FOLDS_VERSION = 'basel'
NORMALIZATION_TYPE = "local_max"



lpps = [load_lesions(PATCH_SIZE, deformed = deform) for deform in LESIONS]

print("[INFO] Loading DA folds...")

folds_data = get_folded_data(lpps, 
                              data_augmentation = "all", 
                              normalization_type = NORMALIZATION_TYPE,
                              contrasts=NETWORK_CONTRASTS,
                              folds_version=FOLDS_VERSION)

print("\n[INFO] Loading NON-DA folds...")

folds_data_test = get_folded_data(lpps, 
                              data_augmentation = "none", 
                              normalization_type = NORMALIZATION_TYPE,
                              contrasts=NETWORK_CONTRASTS,
                              folds_version=FOLDS_VERSION)



train_fold_ncv(NETWORK, FOLD, folds_data, folds_data_test, contrasts=NETWORK_CONTRASTS, 
               da_strategy=DA_ONLINE_STRATEGY, net_name=NETWORK_NAME, epochs=EPOCHS)

