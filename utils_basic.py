import numpy as np

def crop_lesions_from_image(image, lesions, patch_size, is_mask=False):
    """
        Function that, given the center of mass of all lesions, it extracts them from the "image" (cropping).
        The parameter "lesions" must be the return value of function "extract_lesions_from_mask", or the
        "json" of the patient folder (where the center of each lesion is stored).
        
        Returns:
            lesions_patches: dictionary where KEY is lesion_id and VALUE is the lesion as a numpy array
        
    """
    lesions_patches = {}
    lesions_bboxes = {}
    extr = np.round((patch_size / 2).astype(int))
    
    # we first globally normalize the image
    if not is_mask:
        image = (image - np.min(image))/(np.max(image) - np.min(image))
        
    for lesion_id in sorted(lesions.keys()):
        com = lesions[lesion_id]["center"]
        
        min_ = (com - extr).astype(int)
        max_ = (com + extr).astype(int)
        
        if (min_ >= (0,0,0)).all() and (max_ < image.shape).all():
            lesions_patches[str(lesion_id)] = image[min_[0]:max_[0], min_[1]:max_[1], min_[2]:max_[2]].astype('float32')
    
    return lesions_patches