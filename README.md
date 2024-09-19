# SegLib
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

SegLib: a small library for the training and inference of segmentation models.

## Project Structure
This project is structured as follows:
- `docker`: files related to the environment used for training and testing the model. Includes files to build a Docker image or a Singularity image;
- `notebooks`: contains the Jupyter notebooks used during experimentation;
- `scripts`: contains the executable files for the repository, such as the ones used for: training the models, inference, evaluation, etc.;
- `seg_lib`: library containing all of the shared source code that is used by the scripts (built using Poetry, more information on the `pyproject.toml` file).

## Installation
In order to use the training and fusion scripts, please have Python 3.10 installed on your machine. We strongly recommend that the user creates a [conda](https://docs.anaconda.com/miniconda/miniconda-install/) environment for such purpose:
```bash
# create the environment
conda create -n seg_lib_env python==3.10
# activate the environment
conda activate seg_lib_env
```

Later on, our library needs to be installed. Such action may be perform by using `pip` or by building it from scratch. Both processes are described in the following subsections.

### Using Pip
The library can be installed by executing the following [pip](https://pypi.org/project/seg-lib/) command:
```bash
# install the latest version
pip install seg-lib
```

### Using the Local Build
To build the library locally, please follow the steps as described:

```bash
# install poetry for dependency management and library building
pip install poetry
# build the library using poetry
poetry build
# install the built wheel file
pip install dist/seg_lib-1.0.3-py3-none-any.whl
```

If all of the steps were followed correctly, everything should be set up.
The pre-built `wheel` file may be also found in the `dist/` folder, contained in this repository. 

## Data Input Format
To run the training/evaluation scripts for SAMUS and Segmentation models, first refer to the arguments listed in `scripts/samus/train.py`. Each argument has a brief description and most of their default values are already set. After running the experiments, all of the outputs (including the best model's weights, logs, etc.) may be found in the path described in `--output_path`.

It is worth mentioning that the scripts expect that the data are described in a CSV file, contained in a `metadata` folder existing in `--data_path`. In order to set the data in the desired format, please follow the steps described as follows:

1. create a folder called `metadata`, such as: `/path/to/your/data/metadata`
2. place a CSV file `{DATASET_SNAME}.csv` inside the matadata path
3. create the headers to the file and fill it with the information about each pair of image sample and label mask, as:
    a. split: dataset split, used to filter the data during training and testing. Should be: `train`, `val` or `test`
    b. class_id: integer describing the mask class identifier. If the problem deals with binary masking, it is assumed that this value is 1
    c. subset: name of the subset wheret the image files are stored. Should be placed inside `--data_path`
    d. img_name: file name of the input image source. Should be stored in `{data_path}/{subset}/img/{img_name}`
    e. label_name: file name of the refering mask image. Should be stored in `{data_path}/{subset}/label/{label_name}`


By the end of this process, it is expected that the metadata are created in the format:
```
   split  class_id         subset                    img_name                     label_name
0  train         1  ribs_original  VinDr_RibCXR_train_000.png  VinDr_RibCXR_train_000_R1.bmp
1  train         1  ribs_original  VinDr_RibCXR_train_000.png  VinDr_RibCXR_train_000_R2.bmp
2  train         1  ribs_original  VinDr_RibCXR_train_000.png  VinDr_RibCXR_train_000_R3.bmp
3  train         1  ribs_original  VinDr_RibCXR_train_000.png  VinDr_RibCXR_train_000_R4.bmp
4  train         1  ribs_original  VinDr_RibCXR_train_000.png  VinDr_RibCXR_train_000_R5.bmp
```

and that the data follow the structure:
```
- metadata
    |_ dataset_descriptor.csv
- subset_1
    |- img
        |- img_name_1.png
        |- img_name_2.png
        |- ...
        |_ img_name_n.png
    |- label
        |- img_name_1.bmp
        |- img_name_2.bmp
        |- ...
        |_ img_name_n.bmp
- subset_n
    |- img
        |- img_name_1.png
        ...
    |- label
        |- img_name_1.bmp
        ...
- ...
```

Some samples of datasets may be found at `data/`.

## License
This repository is licensed under Apache 2.0.

It is advised to also check [SAM](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE), [SAMUS](https://github.com/xianlin7/SAMUS/blob/main/LICENSE) and [VinDr-RibCXR](https://github.com/vinbigdata-medical/MIDL2021-VinDr-RibCXR) licenses on their original repositories.