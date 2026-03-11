# Information about Target datasets:

Each of the datasets has an associated Pydantic model associated with it defined in `src/model_ranking/dataclass.py`. A pydantic model is very similar/equivalent to a standard python dataclass and each model contains all the neccessary meta data required to run the methods from the repo with that dataset. The models are named with the following nomenclature, {Dataset_Name}TargetConfig. Each model come with prefilled arguments in the various field for convenience and to provide the same setting neccessary reproduce the paper results. :exclamation Importantly :exclamation relative paths to data locations are specified in the varoius dataloader meta data, it is required to arrange the datasets with the same file structure, or to update the relevent arguments, for the code to run.

All the aforementioned Target Dataset configs inherit from TargetDatasetConfigBase and have the following standard structure.

**class TargetDatasetConfigBase():**
- **name**: Name of dataset taken from select supported `Literal`
- **loader**: Defining the meta data for the inferance Dataloader Class, there are 4 implemented classes.
- **predictor_semantic**: Defining the meta data for the semantic segmentation predictor class.
- **eval_dataloader_instance**: Defining the meta data for the instance segmentation predictor class.
- **eval_dataloader_semantic**: Defining the meta data for the semantic segmentation evaluation Dataloader Class.
- **eval_dataloader_instance**: Defining the meta data for the instance segmentation evaluation Dataloader Class.
- **consis_dataloader_semantic**: Defining the meta data for the semantic segmentation consistency score Dataloader Class.
- **consis_dataloader_instance**: Defining the meta data for the instance segmentation consistency score Dataloader Class.
- **filter_results**: Defining class to filter results if desired.


# Target Dataset Directory Structure
In the code it is assumed that each target dataset, identified by name, is stored with a particular directory structure. In this way the correct paths can be identified and the correct Dataloaders and predictors utilised. Below is a summary of the assumed correct directory structure per dataset.


## Mitochondria (EM)

### EPFL

https://www.epfl.ch/labs/cvlab/data/data-em/ 

EPFL is a volume EM dataset containing semantic labels of mitochondria. The dataset consists of train.h5, test.h5 and val.h5 files that each contain 3D datavolumes. The raw and label datavolumes in each file have keys "raw" and "labels" respectively. The pixels have been resized to the same scale as the Hmito/Rmito dataset.

#### Dataset Preprocessing: 
The original EPFL dataset was comprised of three h5 file with data volume dimensions a shown below, these were resized so that the pixel sizes matched the Hmito/Rmito Datasets, helper script to resize data can be found in `data/helper_scripts/pixel_resize.py`.

Original --> Resized: 
train.h5
- label: Dataset {165, 768, 1024}  ---> Dataset {165, 480, 640}
- raw: Dataset {165, 768, 1024}    ---> Dataset {165, 480, 640}

val.h5
- label: Dataset {40, 768, 1024}   ---> Dataset {40, 480, 640}
- raw: Dataset {40, 768, 1024}     ---> Dataset {40, 480, 640}

test.h5
- label: Dataset {125, 768, 1024}  ---> Dataset {125, 480, 640}
- raw: Dataset {125, 768, 1024}    ---> Dataset {125, 480, 640}


#### Directory Structure

```
EPFL/
├── test.h5
├── train.h5
└── val.h5
```

### Hmito 

https://mitoem.grand-challenge.org/

Hmito is a volume EM dataset containing semantic labels of mitochondria. The dataset consists of train_converted.h5, test_converted.h5 and val_converted.h5 files that each contain 3D datavolumes. The raw and label datavolumes in each file have keys "raw" and "labels" respectively. The volumes have been converted to only contain semantic mitochondria labels.

#### Dataset Preprocessing

The original Hmito dataset is a single data volume with size {500, 4096, 4096}. The volume was split into test, train and val sub volumes by slicing the original volume. Code to assist can be found at `data/helper_scripts/split_data.py`

- train --> [[0, 300], [0, 4096], [0, 4096]]
- val --> [[300, 350], [0, 4096], [0, 4096]]
- test --> [[350, 500], [0, 4096], [0, 4096]]

After slicing the original volume into three distinct subsets the labels nwere converted from instance mitochondria labels to semantic mitochondria labels. Code to assist can be found at `data/helper_scripts/convert_to_binary_labels.py`.


#### Directory Structure

```
Hmito/
├── test_converted.h5
├── train_converted.h5
└── val_converted.h5
```

### Rmito 

https://mitoem.grand-challenge.org/

Rmito is a volume EM dataset containing semantic labels of mitochondria. The dataset consists of train_converted.h5, test_converted.h5 and val_converted.h5 files that each contain 3D datavolumes. The raw and label datavolumes in each file have keys "raw" and "labels" respectively. The volumes have been converted to only contain semantic mitochondria labels.

#### Dataset Preprocessing
The original Rmito dataset is a single data volume with size {500, 4096, 4096}. The volume was split into test, train and val sub volumes by slicing the original volume. Code to assist can be found at `data/helper_scripts/split_data.py`

- train --> [[0, 300], [0, 4096], [0, 4096]]
- val --> [[300, 350], [0, 4096], [0, 4096]]
- test --> [[350, 500], [0, 4096], [0, 4096]]

After slicing the original volume into three distinct subsets the labels nwere converted from instance mitochondria labels to semantic mitochondria labels. Code to assist can be found at `data/helper_scripts/convert_to_binary_labels.py`.

#### Directory Structure

```
Rmito/
├── test_converted.h5
├── train_converted.h5
└── val_converted.h5
```

### VNC

https://connectomics.hms.harvard.edu/adult-drosophila-vnc-tem-dataset-female-adult-nerve-cord-fanc

VNC is a volume EM dataset containing semantic labels of mitochondria. The dataset consists of a single h5 file that contains raw and label 3D datavolumes. The raw and label datavolumes have keys "resized_raw" and "resized_labels" respectively. The pixels have been resized to the same scale as the Hmito/Rmito dataset. There is only a single volume and due to data constraints we trained the VNC model on the entire dataset and so no VNC source model -> VNC target dataset transfer should be performed.

#### Dataset Preprocessing

The original VNC dataset comprises a single h5 file with data volume dimensions as shown below, these were resized so that the pixel sizes matched the Hmito/Rmito Datasets, helper script to resize data can be found in `data/helper_scripts/pixel_resize.py`.

Original --> Resized
- label: Dataset {20, 1024, 1024}   --->  resized_labels: Dataset {20, 589, 589}
- raw: Dataset {20, 1024, 1024}     ---> resized_raw: Dataset {20, 589, 589}   


#### Directory Structure

```
VNC/
└── data_labeled_mito.h5
```


## Nuclei (Light Microscopy)
### BBBC039 

https://bbbc.broadinstitute.org/BBBC039

BBBC039 is fluorescent nuclei dataset, with instance annotations of each nuclei. Images are stored as individual 2D .tif files and their corresponding labels as individual 2D .png files. All images/labels are stored in a single directory and can be identified by a unique filename. Within `data/BBBC039` is provided test.txt, train.txt and val.txt which contain sets of line seperated filenames to identify test, train, val split of dataset.

Dataset: TIF_txt_Dataset


#### Directory Structure

```
BBBC039/
├── images/
│   ├── IXMtest_A02_s1_w1051DAA7C-7042-435F-99F0-1E847D9B42CB.tif
│   ├── ...
│   └── IXMtest_P24_s9_w13AC6C03C-E8D7-4A23-B649-514BB4052F52.tif
├── instance_annotations/
│   └── instance_labels/
│       ├── IXMtest_A02_s1_w1051DAA7C-7042-435F-99F0-1E847D9B42CB.png
│       ├── ...
│       └── IXMtest_P24_s9_w13AC6C03C-E8D7-4A23-B649-514BB4052F52.png
├── test.txt
├── train.txt
└── val.txt
```

### DSB2018 

https://bbbc.broadinstitute.org/BBBC038

DSB2018 is fluorescent nuclei dataset, with instance annotations of each nuclei. Images are stored as individual 2D .tif files and their corresponding labels likewise stored as individual 2D .tif files. Only a test set was used as no DSB2018 model was trained.

Dataset: Standard_TIF_Dataset

#### Dataset Preprocessing
Subset of the original DSB2018 dataset was formed that only contains the fluorescent data.

#### Directory Structure

```
dsb2018_fluorescence/
└── test/
    ├── images/
    │    ├── 0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe.tif
    │    ├── ...
    │    └── ff599c7301daa1f783924ac8cbe3ce7b42878f15a39c2d19659189951f540f48.tif
    └── masks/
        ├── 0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe.tif
        ├── ...
        └── ff599c7301daa1f783924ac8cbe3ce7b42878f15a39c2d19659189951f540f48.tif
```


### Go-Nuclear

https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1026?query=S-BIAD1026

Go-Nuclear is fluorescent nuclei dataset, with instance annotations of each nuclei. The dataset consists of multiple h5 files, containing 3D "raw/clear" data volumes and a "label/gold" data volumes for the raw and label data respectively. All the h5 file are in a single directory.  Within `data/Go-Nuclear` is provided test.txt, train.txt, which contain sets of line seperated filenames to identify test vs train split of the dataset.

Dataset: StandardHDF5Dataset

#### Dataset Preprocessing
The following train, test, val split was used. Train - (1135.h5, 1136.h5, 1137.h5), Test - (1170.h5), Val - (1139.h5). The ids are detailed in `data/Go-Nuclear/train.txt`, `data/Go-Nuclear/test.txt` and `data/Go-Nuclear/val.txt`.

#### Directory Structure

```
Go-Nuclear/
└── 3d_all_in_one
    ├── 1135.h5
    ├── ...
    └── 1170.h5
```

### HeLaNuc

https://rodare.hzdr.de/record/3001

HeLaNuc is fluorescent nuclei dataset, with instance annotations of each nuclei. Images are stored as individual .tif files and their corresponding labels as individual .tif files. Images and labels are split into train, test and val directories.

Dataset: HeLaNuc_Dataset

#### Dataset Preprocessing
The train, test, val split used is specifed by the train.txt, test.txt and val.txt files located in `data/HeLaNuc`.

#### Directory Structure

```
HeLaNuc/
├── train/
│   ├── images/
│   │   ├── 2408.tif
│   │   ├── ...
│   │   └── 2675.tif
│   └── nuclei_masks/
│       ├── 2408.tif
│       ├── ...
│       └── 2675.tif
├── test/
│   ├── images/
│   └── nuclei_masks/
└── val/
    ├── images/
    └── nuclei_masks/
```

### Hoechst 

https://zenodo.org/records/6657260

Hoechst is fluorescent nuclei dataset, with instance annotations of each nuclei. Images are stored as multiple .tif files each of which contains a 3D stack of data, the corresponding labels are stored equivalently. Images and labels are split into train, test and val (development_nuclei) directories.

Dataset: Hoechst_Dataset

#### Dataset Preprocessing
The train, test, val split used is specifed by the train.txt, test.txt and dev.txt files located in `data/Hoechst`.


#### Directory Structure

```
Hoechst/
├── test_nuclei/
│   ├── annotations/
│   │   ├── MFGTMPcx7_170702000001_G14f03d0_objects.png
│   │   ├── ...
│   │   └── MFGTMPcx7_170803210001_J12f29d0_objects.png
│   └── images/
│       └── png/
│           ├── MFGTMPcx7_170702000001_G14f03d0_objects.png
│           ├── ...
│           └── MFGTMPcx7_170803210001_J12f29d0_objects.png
├── development_nuclei/
│   ├── annotations/
│   └── images/
└── training_nuclei/
    ├── annotations/
    └── images/
```

### S-BIAD634

https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD634?query=S-BIAD634

S-BIAD634 is fluorescent nuclei dataset, with instance annotations of each nuclei. Images are stored as individual .tif files and their corresponding labels are likewise stored as individual .tif files. All images/labels are stored in a single directory and can be identified by a unique filename. Within `data/S-BIAD634` is provided test.txt, train.txt, which contain sets of line seperated filenames to identify test vs train split of the dataset.


Dataset: TIF_txt_Dataset

#### Directory Structure

```
S-BIAD634/
├── dataset/
│   ├── rawimages
│   │   ├── Ganglioneuroblastoma_0.tif
│   │   ├── ...
│   │   └── otherspecimen_9.tif
│   └── groundtruth
│       ├── Ganglioneuroblastoma_0.tif
│       ├── ...
│       └── otherspecimen_9.tif
├── test.txt
└── train.txt
```

### S-BIAD895

https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD895

S-BIAD895 is fluorescent nuclei dataset, with instance annotations of each nuclei. Images are stored as individual 2D .tif files and their corresponding labels are likewise stored as individual 2D .tif files. Images and labels are split into Train, Test directories. :exclamation Importantly :exclamation you may notice that the inference/evaluation loaders use the `Train` subset during prediction and evaluation. This is not a mistake, but was decided as the `Test` set was too small to get a reliable consistency measure. Hence importantly we never perform S-BIAD895 source model -> S-BIAD895 Target Dataset transfer in order to avoid data leakage.

Dataset: Standard_TIF_Dataset

#### Dataset Preprocessing
The train, test split used is specifed by the train.txt, test.txt files located in `data/S-BIAD895`.

#### Directory Structure

```
S-BIAD895/
└── ZeroCostDL4Mic/
    └── Stardist_v2/
        └── Stardist/
            ├── Train/
            │    ├── Raw/
            │    │    ├── cell migration R1 - Position 0_XY1562686096_Z0_T00_C1-1-image1.tif
            │    │    ├── ...
            │    │    └── migration r2 - Position 61_XY1562769118_Z0_T00_C1-1.tif
            │    └── Masks/
            │        ├── cell migration R1 - Position 0_XY1562686096_Z0_T00_C1-1-image1.tif
            │        ├── ...
            │        └── migration r2 - Position 61_XY1562769118_Z0_T00_C1-1.tif
            └── Test/
                ├── Raw/
                │    └── ...
                └── Masks/
                    └── ...
```

### S-BAID1196 (SELMA3D) 

https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1196?query=S-BIAD1196


S-BIAD1196 is fluorescent nuclei dataset, with instance annotations of each nuclei. The dataset consists of h5 files, containing 3D "raw" data volumes and a "label" data volumes. Images and labels are split into train, test and val directories.

Dataset: StandardHDFDataset

#### Dataset Preprocessing
The train, test, val split used is specifed by the train.txt, test.txt and val.txt files located in `data/S-BIAD1196`.

#### Directory Structure

```
S-BIAD1196/
└── SELMA3D_training_annotated
    └── shannel_cells
        └── h5
            ├── test
            │   ├── patchvolume_009.h5
            │   ├── ...
            │   └── patchvolume_011.h5
            ├── train
                └── ...
            └── val
                └── ...
```

### S-BIAD1410 

https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1410?query=S-BIAD1410

S-BIAD1410 is fluorescent nuclei dataset, with instance annotations of each nuclei. The dataset consists of  multiple .tif files each of which contains a 3D stack of data, the corresponding labels are stored equivalently. Images and labels are split into train, test directories.

Dataset: S_BIAD1410_Dataset

#### Dataset Preprocessing
The train, test  split used is specifed by the train.txt, test.txt files located in `data/S-BIAD1410`.

#### Directory Structure

```
S-BIAD1410/
└── cardioblast_nuclei/
    ├── cardioblast_nuclei_test
    │   ├── cardioblast_nuclei_20200121_e3/
    │   │   ├── cardioblast_nuclei_20200121_e3_mask.tif
    │   │   └── cardioblast_nuclei_20200121_e3.tif
    │   ├── ...
    │   └── cardioblast_nuclei_20220828_e3/
    │       ├── cardioblast_nuclei_20220828_e3_mask.tif
    │       └── cardioblast_nuclei_20220828_e3.tif
    └── cardioblast_nuclei_train
        └── ...
```

## Cells (Light Microscopy)
### FlyWing

https://elifesciences.org/articles/57613

FlyWing is a Fluorescent cell dataset, with instance cell labels. The dataset consists of mulitple h5 files that each contain 3D data volumes. The keys to the raw and label volumes are "volumes/raw" and "volumes/labels/cells_with_ignore" respectively. Train, test and val images are stored in seperate directories.

Dataset: StandardHDF5Dataset

#### Dataset Preprocessing
The train, test and val splits used are specifed by the train.txt, test.txt and val.txt files located in `data/FlyWing`.

#### Directory Structure

```
FlyWing/
└── GT/
    ├── test/
    │   ├── per03.h5
    │   └── pro03.h5
    ├── train/
    │   └── ...
    └── val/
        └── ...
```

### Ovules

https://elifesciences.org/articles/57613

Ovules is a Fluorescent dataset, with instance cell labels. The dataset consists of mulitple h5 files that each contain 3D data volumes. The keys to the raw and label volumes are "raw" and "label_with_ignore" respectively. Train, test and val images are stored in seperate directories.

Dataset: StandardHDF5Dataset

#### Dataset Preprocessing
The train, test and val splits used are specifed by the train.txt, test.txt and val.txt files located in `data/Ovules`.

#### Directory Structure

```
Ovules/
└── GT2x/
    ├── test/
    │   ├── N_294_final_crop_ds2.h5
    │   ├── ...
    │   └── N_593_final_crop_ds2.h5
    ├── train/
    │   └── ...
    └── val/
        └── ...
```

### PNAS

https://pubmed.ncbi.nlm.nih.gov/27930326/ 

PNAS is a Fluorescent dataset, with instance cell labels. The dataset consists of mulitple h5 files that each contain 3D data volumes. The keys to the raw and label volumes are "raw" and "label" respectively. Train, test and val images are stored in seperate directories.

Dataset: StandardHDF5Dataset

#### Dataset Preprocessing
The train, test and val splits used are specifed by the train.txt, test.txt and val.txt files located in `data/PNAS`.

#### Directory Structure

```
PNAS/
├── test/
│   ├── 4hrs_plant1_trim-acylYFP.h5
│   ├── ...
│   └── 80hrs_plant13_trim-acylYFP.h5
├── train/
│   └── ...
└── val/
    └── ...
```