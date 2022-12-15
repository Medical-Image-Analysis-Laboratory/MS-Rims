
import numpy as np
from config import *
import scipy.ndimage as snd
from tqdm import tqdm
import elasticdeform


def __recenter_and_crop_lesion(lesion_id, lesion, goal_size, recenter):
    key_mask = None
    if "MASK" in lesion.keys():
        key_mask = "MASK"
    elif "MASK_SK" in lesion.keys():
        key_mask = "MASK_SK"
    elif recenter:
        raise Exception("[ERROR] MASK needed to center lesion and crop it.")

    if recenter:
        mask = (lesion[key_mask] == int(lesion_id)).astype(int)
        com = np.round(snd.measurements.center_of_mass(mask))
    else:
        shape = lesion[list(lesion.keys())[0]].shape
        com = [int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)]
    
    extr = np.round((goal_size / 2).astype(int))
    min_ = (com - extr).astype(int)
    max_ = (com + extr).astype(int)
    
    for c in lesion.keys():
        if (min_ >= (0,0,0)).all() and (max_ < lesion[c].shape).all():
            lesion[c] = lesion[c][min_[0]:max_[0], min_[1]:max_[1], min_[2]:max_[2]]
        else:
            raise Exception(f"Can't crop a region of desired size ({goal_size}) around the center ({com})")
    
       
    if (lesion[list(lesion.keys())[0]].shape != goal_size).any():
        print("CAREFUUUUUUL, something failed and it should not have!!")
    return lesion          

def deform_and_crop_lesion(lesion_id, lesion, goal_size, recenter=True):
    contrasts = list(lesion.keys())
    to_deform = [lesion[c] for c in contrasts]

    # apply deformation with a random 3 x 3 grid
    orig_size = to_deform[0].shape
    diff = orig_size - goal_size
    # We interpolate all contrasts except the mask
    orders = [3 if c != "MASK" and c != "MASK_SK" else 0 for c in contrasts]
    
    deformed = elasticdeform.deform_random_grid(to_deform, sigma=1, points=3, order=orders)
    
    deformed_lesion = {}
    for i in range(len(contrasts)):
        deformed_lesion[contrasts[i]] = deformed[i]
       
    try:
        final_lesion = __recenter_and_crop_lesion(lesion_id, deformed_lesion, goal_size, recenter)
    except Exception  as ex:
        print(ex)
        return None
    
    return final_lesion


    
            

