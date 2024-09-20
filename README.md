# SegLib: A Segmentation Library for 
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

`SegLib` was developed in a partnership between researchers from the [Università degli Studi di Padova (Italy)](https://www.unipd.it/) and the [Universidade Estadual de Maringá (Brazil)](https://www.uem.br/). It provides an interface designed to support the development of segmentation models such as PolyPVT (referred to here as SegPVT), CAFE-net, SAM, SAMv2, SAMUS, SAM-Med2D, and others. The library includes these models in PyTorch format, along with standard training classes, predictors (for SAM-like architectures), data loaders, data augmentation pipelines, and more.

The library is currently being used for the development of various projects, which can be found under the `projects` subpath. More details about these projects can be found in the following subsections.

## Project Structure
This project is structured as follows:
- `data`: A sample dataset organized in the expected input format for training and evaluating the described projects.
- `docker`: Files related to the environment used for training and testing the model, including scripts to build a Docker image or a Singularity image.
- `projects`:  Scripts for the projects developed using the library.
- `seg_lib`: The library containing all shared source code used by the scripts (built using Poetry; more details can be found in the `pyproject.toml` file);
- `tools`: Standalone tools..

## Projects
Under the `projects/` path, you can find the source code used in our developed articles. Please refer to the `README.md` file in each individual project for more details on how to execute them. A brief description of each project is provided below.

### sample-size-for-training-and-testing
Scripts developed for the book chapter:
- **Title**: Sample Size for Training and Testing: Segment  Anything models and supervised approaches
- **Authors**: Daniela Cuza, Carlo Fantozzi, Loris Nanni, Daniel Fusaro, Gustavo Zanoni Felipe & Sheryl Brahnam
- **Book Title**: Advances in Intelligent Healthcare Delivery and Management
- **Publisher**: Springer
- **Year**: 2024

**Abstract**
> The problem of determining the minimum amount of data required to train and test an artificial intelligence model has received substantial attention in the literature. In this chapter, we first review key concepts on the topic, then we survey selected theoretical and experimental results from the open literature, and in the end we present, as a case study, experiments we performed ourselves on the semantic segmentation of radiology images. A discussion from both a theoretical and an experimental point of view is required because the two approaches have complementary insights to offer. Theory provides general guidelines to avoid pitfalls during all phases of design: data collection, model design, training, and testing. Experimental results show what the current state of the art is in terms of performance and provide practical advice on which techniques have proven to be the most effective; for a more comprehensive study, we tested both supervised and zero-shot segmentation approaches, such as the “Segment Anything Model” (better known as SAM).

For details on how to execute the sources, please refer to the README.md located at `projects/sample-size-for-training-and-testing/README.md`. Also, the full citation may be found at `projects/sample-size-for-training-and-testing/CITATION.bib`.

## Installation
To install the library, ensure that Python 3.10 is set up on your machine. We strongly recommend that the user creates a [conda](https://docs.anaconda.com/miniconda/miniconda-install/) environment for such purpose:

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
Our dataloaders expects that the data are described in a CSV file, contained in a `metadata` folder existing in the data path. In order to set the data in the desired format, please follow the steps described as follows:

1. create a folder called `metadata`, such as: `/path/to/your/data/metadata`
2. place a CSV file `{DATASET_NAME}.csv` inside the matadata path
3. create the headers to the file and fill it with the information about each pair of image sample and label mask, as:

    - split: dataset split, used to filter the data during training and testing. Should be: `train`, `val` or `test`.
    - class_id: integer describing the mask class identifier. If the problem deals with binary masking, it is assumed that this value is 1.
    - subset: name of the subset where the image files are stored. Should be placed inside `/path/to/your/data/{subset}`.
    - img_name: file name of the input image source. Should be stored in `/path/to/your/data/{subset}/img/{img_name}`.
    - label_name: file name of the refering mask image. Should be stored in `/path/to/your/data/{subset}/label/{label_name}`


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