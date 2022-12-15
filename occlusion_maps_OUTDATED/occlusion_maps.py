

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
import os
from scipy import ndimage
import archs.cnn_2 as cnn2
import archs.cnn_7 as cnn7
from utils import get_folds_indexes
from config import *


def normalize(image):
    image = image.copy()
    for idx in range(image.shape[3]):
        image[:,:,:,idx] = normalize_patch(image[:,:,:,idx], "local_max")
    return image

def preprocess(image, i, j, k):
    size = 2
    image = image.copy()
    # we hide all channels
    for idx in range(image.shape[3]):
        image[:,:,:,idx] = normalize_patch(image[:,:,:,idx], "local_max")
        image[size*i:size*(i+1), size*j:size*(j+1), size*k:size*(k+1), idx] = -1
    #image[i:i+size, j:j+size, k:k+size] = -1
    return image

def get_occlusion_map(patient, image):
    if len(image.shape) == 3:
        image = image.reshape(28,28,28,1)
    
    cnn = cnn7
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    patch_size = width, height, depth = [28, 28, 28]
    name = "07_unet_DA3_DEF" #"02_unet_NCV_DA3"
    num_epochs = [12, 13, 11, 12, 9] #[18, 30, 24, 19, 22]
    testing_folds = [fold["testing"] for fold in get_folds_indexes()]
    
    num_fold = 0
    found = False
    while not found and num_fold < len(testing_folds):
        if patient in testing_folds[num_fold]:
            num_epochs = num_epochs[num_fold]
            found = True
        else:
            num_fold += 1
        
    PATH_TESTING = os.path.join(ROOT_DIR, os.path.join("checkpoints", name))
    FILENAME = f"check-{num_epochs}"
    path_testing = os.path.join(os.path.join(PATH_TESTING, str(num_fold)), FILENAME)

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        x, _, _, _, _, _, pred = cnn.get_model_graph(patch_size)
    
    with tf.Session(graph=g) as sess:

        # Initialization
        init = tf.initializers.global_variables()

        sess.run(init)

        saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        saver.restore(sess, path_testing)
        
        raw_pred = sess.run(pred, feed_dict = {x: [normalize(image)]})[0][1]

        occlusion_map = np.array([[[sess.run(pred, feed_dict = {x: [preprocess(image, i, j, k)]})[0][1] 
                           for k in range(image.shape[2]//2)]
                          for j in range(image.shape[1]//2)]
                         for i in range(image.shape[0]//2)])
        
    image = image.squeeze()
    occlusion_map_upsampled = ndimage.interpolation.zoom(occlusion_map, np.array(image.shape)/np.array(occlusion_map.shape, dtype=np.float32), order=3, mode='nearest')
    return raw_pred, 1-occlusion_map_upsampled
