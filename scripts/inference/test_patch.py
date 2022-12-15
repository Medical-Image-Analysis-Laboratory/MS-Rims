import sys
sys.path.append("../../../MS-Rims/")
import os
import numpy as np
from inference import Patch

if __name__ == "__main__":
    t2_st_mag = '/data/T2STAR_MAG_rim_patcth.nii.gz'
    fl = '/data/FLAIR_rim_patcth.nii.gz'
    patch_dict = {'T2STAR_MAG':t2_st_mag, 'FLAIR':fl} 
    patch_dict_2 = {'FLAIR':fl, 'T2STAR_MAG':t2_st_mag} 
    ex_patch = Patch(patch_dict)
    print(ex_patch.contrasts_dict_paths)
    ex_patch.load()
    ord_1 = ex_patch.process_lesion(['T2STAR_MAG', 'FLAIR'])
    
    ex_patch_2 = Patch(patch_dict_2)
    ord_2 = ex_patch_2.process_lesion(['T2STAR_MAG', 'FLAIR'])
    print(sum(sum(ord_1 - ord_2)))
    conc_arr = np.stack([ord_1, ord_2, ord_1], axis=0)
    print(conc_arr.shape)
    