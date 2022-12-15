import sys
sys.path.append("..")
from location import compute_locations
from config import *

compute_locations(datasets = [DATASET_CHUV_ID, ], cpus = 16, replace = False)