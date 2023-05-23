# Tools for Automatic Classification of Rim Lesions in Multiple Sclerosis: Streamline RimNet

This site provides two tools to ease the classification of Paramagnetic Rim Lesions employing **RimNet**, the method proposed in the paper "*RimNet: A deep 3D multimodal MRI architecture for paramagnetic rim lesion assessment in multiple sclerosis*" by Barquero et al. NeuroImage: Clinical (2020).

To boost the use of **RimNet** by clinicians and medical imaging practitioners, we present two independent tools capable of making predictions over patches containing Rim lesions when employing FLAIR and T2* Phase MRI contrasts from the same model presented in the mentioned work.
1. 3D-Slicer plugin: This tool is more appropriate for research and clinical environments where users need to manually inspect images to select suspicious MS manifestations.
2. Dockerized version of RimNet: This tool is more appropriate for processing pipelines as it allows RimNet to be run, thereby obtaining predictions for possible rims by executing a single command.

![Alt Text](https://github.com/Medical-Image-Analysis-Laboratory/MS-Rims/blob/main/StreamlineRIMNET.png)

## 1. Using 3D-Slicer plugin

Within the file `filename`, we include a standalone version of 3D-Slicer with the semi-rimnet plugin which allows:

- Comfortably navigate through the different available MRI contrasts to check for lesions (align images, switch contrasts, crosshair with mouse drag & drop navigation, etc.)
- Pick possible lesions by manual selection
- Pick possible lesions automatically by using a binary mask
- Add annotations and the expert confidence opinion
- Run RimNet to obtain a prediction about the lesion being a rim or not
- Save the data for further analysis

Within `filename`, we include a complete manual, `Instructions.pdf`.

## 2. Using RimNet with Docker

If you want to use only the Docker, just download the `rimnet-basics` file and unzip it wherever is more convenient for you. This file contains two folders:

- `data` folder: where the patches have to be stored and an example of the CSV file needed to run the Docker (see below).
- `models` folder: containing the RimNet model weights inside.

### Pulling the Docker

To pull the RimNet Docker container, run the following command:
```
foo@bar:~$ docker pull ghcr.io/medical-image-analysis-laboratory/rimnet:latest
```

### Running the Docker

Since RimNet is patch-based, before running the Docker containing RimNet, you will need to extract or gather previously extracted patches of size 28x28x28. All patches must be located under the `rimnet-basics/data` directory (to be mounted by the Docker), however, the hierarchy inside the folder is not relevant since the relative path information will be codified in the CSV file.

The CSV file must contain the following header to identify each pair of patches (remember that RimNet is fed by the patches at the same location from the two MRI contrasts, T2*Phase and FLAIR) uniquely:
- `sub_id`: codify the subject ID from where the patch was extracted
- `patch_id`: identify the patch within the subject
- `cont_T2STAR_PHASE`: Points to the relative path inside the patches folder for the patch corresponding with T2* Phase contrast.
- `cont_FLAIR`: Points to the relative path inside the patches folder for the patch corresponding with FLAIR contrast


Besides, you can add more fields for extra information. We provide an example of such CSV file: check `patches_description_example.csv`

Once your data and CSV file are ready, you can run the Docker in the following way:

```
foo@bar:~$ docker run -v /path/to/rimnet-basics/rimnet-basics/data:/data -v /path/to/rimnet-basics/rimnet-basics/models:/models --gpus device=0 ghcr.io/medical-image-analysis-laboratory/rimnet:latest patches_description_example.csv --model binet_phase_flair
```


Make sure to replace `/path/to/rimnet-basics/` with your route as the name of the CSV file previously configured.

After running, the file `predictions_binet_phase_flair_all.csv` will be created, giving a prediction per `patch_id`.



#### People using in part or fully this software, any of both tools, should cite:

[1] G. Barquero et al., Neuroimage Clinical (2020), DOI: 10.1016/j.nicl.2020.102412
[2] J. Najm et al., Zenodo 2023, DOI: 10.5281/zenodo.7962482




# RimNet: A deep 3D multimodal MRI architecture for paramagnetic rim lesions assessment in multiple sclerosis

## How to use?
**IMPORTANT:** all the names of variables used below are inside the file *config.py* unless otherwise specified.

**IMPORTANT 2:** information related to training variables and strategies is stored in [RimNet_Versions](https://docs.google.com/spreadsheets/d/1wla6plWgkqoBFNIsHqFssysvDSYBpeGDz9kHYra-njs/edit?usp=sharing).

### Installation
1. Go to *config.py* and update:
    - PATH_DATA: path to the folder that contains Basel and CHUV (and NIH) datasets.
    - MODELS_LOAD_FROM: path where checkpoints for trained models are saved.
    - MODELS_FIGS_SAVE_TO: path where figures generated in the analysis notebook will be saved.
2. Update variables AVAILABLE_DATASETS and AVAILABLE_DATASETS_ROOTS with the datasets included (remove NIH if not used).

### Pre-processing and patch generation

#### V1 - From segmentations
In folder */scripts/preprocessing/*, execute:

1. Extraction of patches of size PATCH_SIZE using the segmentations (derivatives/rim_annotations). They are saved in the derivatives folder *lesions_XX_XX_XX* where XX is the PATCH_SIZE.

    ```
    python extract_lesions.py
    ```

2. Extraction of bigger patches of size PATCH_SIZE_DEFORMATIONS using the segmentations (derivatives/rim_annotations). They are saved in the derivatives folder *lesions_XX_XX_XX* where XX is the PATCH_SIZE_DEFORMATIONS.

    ```
    python generate_deformations_1.py
    ```

3. Deformation of patches extracted in step 2, application of an elastic deformation with the seed passed as argument and its cropping to size of PATCH_SIZE. They are saved in the derivatives folder *lesions_XX_XX_XX_DEF-Y* where XX is the PATCH_SIZE and Y is the seed.

    ```
    python generate_deformations_2.py 1
    python generate_deformations_2.py 2
    python generate_deformations_2.py 3
    ```

4. Checks lesions that need to be cleaned according to the exclusion criterias explained in the paper (NeuroImage: Clinical), including: lesion too small, too big, rim intrusion and air artifact. Lesions too small are the only ones excluded in both training and testing. The others are only excluded for training. The json files inside each patient's *lesions_XX_XX_XX* will be updated with the exclusion flag and the criteria used for its exclusion.

    ```
    python apply_cleaner.py
    ```

#### V2 - Autosplitting of lesions

No pre-processing is needed. The training/testing scripts will work with the json files generated by */utilities/split_lesions_morphology.ipynb* or */utilities/split_lesions_with_pmaps.ipynb*. Those json files consists of the centers and the rim+/- ground truth of all the lesions, previously retrieved.


### Training
Execute */scripts/train/train.py* for V1 and */scripts/train/train_autosplit.py* for V2. In both modes, you need to specify the following variables: NETWORK_CONTRASTS, NETWORK (architecture, which has to be consistent with the number of contrasts fed as input), LESIONS (if you want to use deformed versions, specify their seeds), NETWORK_NAME (name of the folder that will be created to store checkpoints), DA_ONLINE_STRATEGY (see [RimNet_Versions](https://docs.google.com/spreadsheets/d/1wla6plWgkqoBFNIsHqFssysvDSYBpeGDz9kHYra-njs/edit?usp=sharing)), FOLDS_VERSION (data to use in the training, see *get_folds_structure* function in *utils.py*) and NORMALIZATION_TYPE (see [RimNet_Versions](https://docs.google.com/spreadsheets/d/1wla6plWgkqoBFNIsHqFssysvDSYBpeGDz9kHYra-njs/edit?usp=sharing)). For V2, the SPLIT_VERSION has to be specified.

```
python /scripts/train/train.py #NUM_FOLD_TO_TRAIN
```

### Testing
For V1, execute */scripts/test/test.py* and, for V2, execute */scripts/test/test_segm.py*. Please, mind the SPLIT_VERSION variable for the latter.

- network_name: the same it was specified during the training.
- network: network used for training.
- folds_version: folds to use for testing. Can be different from the training, depending on what is the goal of the testing (could test over the training set, for example, or over unseen data).
- ensemble: if probabilities should be averaged across different fold models.

This script will generate a *csv* file with all the results, that can be analyzed in the */testing/analysis.ipynb* notebook.

**IMPORTANT**: the name of the *csv* file is important to know where it comes from. *NameOfTheModel-TestingFold-ENS.csv*, where ENS appears if the predictions of the several models were averaged working as an ensemble of classifiers.

## How to add new sequences

1. Add it to the dataset folder, as raw or derivative.
2. Add it to *config.py* file, in variable CONTRASTS, with the corresponding attributes.
3. If you want to use it as an input for the network, add the key in PURE_CONTRASTS (inside the if not WORKING_MODE_SKULL_STRIPPED clause).
4. If working with mode V1, re-run the installation to extract the patches from the new sequence.


## How to add new patients

Just add them in the corresponding dataset and modify the folds (see *get_folds_structure* function in *utils.py*) to include them in the training/testing.

## How to add new datasets

Add them in AVAILABLE_DATASETS and AVAILABLE_DATASETS_ROOTS in *config.py*.

---

## Dataset - Derivatives available

- **expert_annotations**: rim+ slices or volumes manually annotated by P.M. or M.A.
- **freesurfer_segmentation**: freesurfer segmentation of the patient, and the version with simplified labels for our project.
- **registrations_to_T2star**: FLAIR, MP(2)RAGE images registered to T2* space. 
- **rim_annotations**: segmentations manually corrected to split rim+ lesions labelled as rim+ (1XXX) or rim- (2XXX).
- **segmentation_probability_maps**: probability maps of the segmentation.
- **segmentation**: segmentation_probability_maps after applying a threshold (0.3).
- **synthetic_mp2rage**: residual data that stayed after trying synthetic MP2RAGE with RimNet.

---

## Code structure

- **root**:
    - config.py: main file, where all the variables used in the project are loaded.
    - deformations.py: functions used to deform the patches generated as offline data augmentation.
    - location.py: functions used to extract the location of the lesions, used for the location analysis.
    - splitlesions_loader.py: auxiliar function to load lesions automatically split.
    - testing.py: functions related to the testing/inference phase.
    - training.py: functions related to the training phase.
    - utils.py and utils_basic.py: functions used all over the project. Functions that load the lesions or their metadata can be found here. Also the function that generates the folds is stored here (needed to change the fold structures).
    - widgets.py: functions to visualize lesions along the 3 dimensions (only working in jupyter-notebook, not in jupyter-lab).
- **archs**: several CNNs architectures used for this project.
- **data_analysis**: notebooks used for lesions analysis (volume, confluent, etc)
- **folds_generation**: genetic algorithm used for the automatic generation of fully balanced (rim+/- and by center) folds.
- **scripts**: scripts for pre-processing, training and testing, useful for the translation to a cluster.
- **testing**: notebooks to evaluate the performance of each model, and correlate it to the decisions of the experts.
- **utilities**: other notebooks used, including registration, link of GT with segmentation, automated splitting of lesions, etc.

---

## Architectures
Only the architectures used most are specified here:
- **monomodal**: the straight-forward simple monomodal CNN.
- **rimnet_bi**: RimNet with 2 inputs, as described in the NeuroImageClinical paper.
- **rimnet**: RimNet with 3 inputs, as described in the MIDL paper.
- **rimnet_bi_plus**: like rimnet_bi with increased complexity in the dense layers. Used to check if the problem with the auto-splitting was given by the number of parameters of the network, which was not.
