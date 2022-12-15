import sys
sys.path.append("../../")
import os
import argparse
import numpy as np
import pandas as pd
import glob

import SimpleITK as sitk

from training import generate_epoch_batches, process_lesion
from config import *

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


class Patch:
    def __init__(self, contrasts_dict_paths:dict):
        """
        just asserts that the contrast exists as their paths
        contrasts_dict_paths: {contrast_1:path_1, contrast_2:path_2,...}
        """
        contrasts = list(CONTRASTS.keys())
        self.contrasts_dict_paths = contrasts_dict_paths
        self.paths_contrasts = list(contrasts_dict_paths.keys())
        paths = list(contrasts_dict_paths.values())
        assert all(contrast in contrasts for contrast in self.paths_contrasts), f"One or more provided contrasts: {self.paths_contrasts} are not among the implemented contrasts {contrasts}"
        
        assert all(os.path.exists(path) and os.path.isfile(path) for path in paths), f"One or more provided paths: {paths} do not exits"
        self.patch_contrasts = {} 

    def load(self):
        for contrast, path in self.contrasts_dict_paths.items():
            patch_array = sitk.GetArrayFromImage(sitk.ReadImage(path))
            assert all(patch_array.shape == PATCH_SIZE), f"The patch size in {path} (patch_array.shape) is not the valid patch size: {PATCH_SIZE}"
            self.patch_contrasts[contrast] = patch_array
        
    def process_lesion(self, contrasts_order:list, normalization="local_max"):
        assert all(contrast in self.paths_contrasts for contrast in  contrasts_order), f"One of the required contrast for proccesing {contrasts_order} does not appear among the patch contrasts {self.paths_contrasts}"
        if len(self.patch_contrasts) == 0:
            self.load()
        return process_lesion(self.patch_contrasts, contrasts_order, normalization=normalization)
   
    @staticmethod
    def get_patches_from_df(pd_df:pd.DataFrame, contrast_cols_contains:str='cont_', root_dir=None):
        """
        pd_df: a daframe with a prefix "cont_" (or whatever in @contrast_cols_contain) for the columns with contrasts 
        """
        contrasts_dicts = pd_df.filter(like=contrast_cols_contains).rename(columns=lambda x: x.replace(contrast_cols_contains,''))
        if root_dir is not None:
            for col in contrasts_dicts:
                 contrasts_dicts[col] = contrasts_dicts[col].apply(lambda x: os.path.join(root_dir, x))
        
        contrasts_dicts = contrasts_dicts.to_dict('records')
        patches = [Patch(patch_dict) for patch_dict in contrasts_dicts]
        return np.stack([patch.process_lesion(contrasts) for patch in patches], axis=0)
        

def generate_random_input(n:int, n_contrasts:int):
    pass



if __name__ == "__main__":
    DEFAULT_DIR = '/data'
    parser = argparse.ArgumentParser(description='Specific script to prepare the data for RIMNet')
    parser.add_argument('dataset_file', type=str, 
                        help='The path to the file with the images routes. CSV file with header: id,flair,mp2rage', 
                        default=os.path.join(DEFAULT_DIR, 'dataset.csv'))
    parser.add_argument('--output', type=str, help='The path to the output', 
                        default=os.path.join(DEFAULT_DIR, 'predictions.csv'), required=False)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64, required=False)
    parser.add_argument('--model', type=str, help='model to employ', default="mono_phase_basel",required=False, 
                        choices=os.listdir(MODELS_LOAD_FROM))
    parser.add_argument('--fold', type=str, help='Choose a specif fold from the training to infer from or all of them for an ensemble prediction',
                        default="all",required=False, 
                        choices=['0','1','2','3','all'])
    
    args = parser.parse_args()
    
    data_df = pd.read_csv(os.path.join(DEFAULT_DIR, args.dataset_file))
    network_name = args.model
    network = NETWORKS[network_name][0]
    contrasts = NETWORKS[network_name][1]
    batch_size = args.batch_size
    ensemble = True if args.fold == 'all' else False
    folds = ['0','1','2','3'] if ensemble else [args.fold]
    output_name, extension = args.output.split('.')
    output_name = output_name+'_'+network_name+'_'+args.fold+'.'+extension
    output_path = os.path.join(DEFAULT_DIR, output_name)
    
    #random_input = np.random.randint(0,200, (128, PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2],len(contrasts)))
    images_input = Patch.get_patches_from_df(data_df, root_dir=DEFAULT_DIR)
    checkpoints_path = os.path.join(PATH_CHECKPOINTS, network_name)
    preds = []
    preds_dct = {}

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        x, y, lr, eval_dict, cost, optimizer, pred = network.get_model_graph(PATCH_SIZE_TRAINING, n_channels = len(contrasts))
        
    with tf.Session(graph=g) as sess:
        # Initialization
        init = tf.initializers.global_variables()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=1)
        # Get required predictions per fold
        for fold in folds:
            checkpoint_index_file = glob.glob(os.path.join(checkpoints_path, fold, '*.index'))[0]
            if os.path.exists(checkpoint_index_file):
                saver.restore(sess, checkpoint_index_file.split('.index')[0])
            else:
                sys.exit('Incorrect checkpoint', checkpoint_index_file)
                
            fold_preds = []
            for iteration, image_batch in enumerate(generate_epoch_batches(images_input, None, batch_size, PATCH_SIZE_TRAINING,
                                                                           da_strategy=None, use_all=True, random=False, mode_inference=True)):
                feed_dict = {x: image_batch}
                predictions,  = sess.run([pred, ], feed_dict=feed_dict)
                fold_preds += [p[1] for p in predictions]
                
            preds += [fold_preds]
            preds_dct['fold_'+fold] = fold_preds
    preds_dct['model'] = network_name
    pd.concat([data_df, pd.DataFrame(preds_dct)], axis=1).to_csv(output_path) 
    
