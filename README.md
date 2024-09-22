# SegLib: A library for the development of segmentation models
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

`SegLib` was developed in a partnership between researchers from the [Università degli Studi di Padova (Italy)](https://www.unipd.it/) and the [Universidade Estadual de Maringá (Brazil)](https://www.uem.br/). It provides an interface designed to support the development of segmentation models such as PolyPVT (referred to here as SegPVT), CAFE-net, SAM, SAMv2, SAMUS, SAM-Med2D, and others. The library includes these models in PyTorch format, along with standard training classes, predictors (for SAM-like architectures), data loaders, data augmentation pipelines, and more.

The library is currently being used for the development of various projects, which can be found under the `projects` subpath. More details about these projects can be found in the following subsections.

## Project Structure
This project is structured as follows:
- `data`: A sample dataset organized in the expected input format for training and evaluating the described projects.
- `docker`: Files related to the environment used for training and testing the model, including scripts to build a Docker image or a Singularity image.
- `projects`:  Scripts for the projects developed using the library.
- `seg_lib`: The library containing all shared source code used by the scripts (built using Poetry; more details can be found in the `pyproject.toml` file).
- `tools`: Standalone tools.

## Projects
Under the `projects/` path, you can find the source code used in our developed articles. Please refer to the `README.md` file in each individual project for more details on how to execute them. A brief description of each project is provided below.

### improving-existing-segmentators-performance
Cloned repository of the article (with an additional notebook used for experimentation):
- **Title**: [Improving Existing Segmentators Performance with Zero-Shot Segmentators](https://www.mdpi.com/1099-4300/25/11/1502)
- **Authors**:  Loris Nanni, Daniel Fusaro, Carlo Fantozzi and Alberto Pretto
- **Journal**: Entropy
- **Publisher**: MDPI
- **Year**: 2023
- **DOI**: https://doi.org/10.3390/e25111502

Please refer to the [original repository](https://github.com/LorisNanni/Improving-existing-segmentators-performance-with-zero-shot-segmentators) for more details. The full citation may be found at `projects/improving-existing-segmentators-performance/CITATION.bib`.

### sample-size-for-training-and-testing
Scripts developed for the book chapter:
- **Title**: [Sample Size for Training and Testing: Segment  Anything models and supervised approaches](https://link.springer.com/chapter/10.1007/978-3-031-65430-5_6)
- **Authors**: Daniela Cuza, Carlo Fantozzi, Loris Nanni, Daniel Fusaro, Gustavo Zanoni Felipe and Sheryl Brahnam
- **Book Title**: Advances in Intelligent Healthcare Delivery and Management
- **Publisher**: Springer
- **Year**: 2024
- **DOI**: https://doi.org/10.1007/978-3-031-65430-5_6

For details on how to execute the sources, please refer to the README.md located at `projects/sample-size-for-training-and-testing/README.md`. Also, the full citation may be found at `projects/sample-size-for-training-and-testing/CITATION.bib`.

## Installation
To install the library, ensure that Python 3.10 is set up on your machine. We strongly recommend that the user creates a [conda](https://docs.anaconda.com/miniconda/miniconda-install/) environment, as in:

```bash
# create the environment
conda create -n seg_lib_env python==3.10
# activate the environment
conda activate seg_lib_env
```

After setting up the correct Python version, the library may be installed by using `pip` or by building it from scratch. Both processes are described in the following subsections.

### Using Pip
The library can be installed by executing the following [pip](https://pypi.org/project/seg-lib/) command:

```bash
# install the latest version
pip install seg-lib
# or install a target version
pip install seg-lib==1.0.4
```

### Using the Local Build
To build the library locally, clone this repository to a local machine and run the following commands from the project's root directory:

```bash
# install poetry for dependency management and library building
pip install poetry
# build the library using poetry
poetry build
# install the built wheel file
pip install dist/seg_lib-1.0.4-py3-none-any.whl
```

If all of the steps were followed correctly, everything should be set up.
The pre-built `wheel` file may be also found in the `dist/` folder, contained in this repository. 

## Data Input Format
Our dataloaders expects that the data are described in a CSV file, contained in a `metadata` folder existing in the data path. In order to set the data in the desired format, please follow the steps described as follows:

1. create a folder called `metadata`, such as: `/path/to/your/data/metadata`
2. place a CSV file `{dataset_name}.csv` inside the matadata path
3. fill the CSV file with the required information regarding the data samples. It should present at least the following columns:

    - **split**: dataset split, used to filter the data during training and testing. Should be: `train`, `val` or `test`.
    - **class_id**: integer describing the mask class index. If the problem deals with binary masking, it is assumed that this value is set to 1.
    - **subset**: name of the set where the image files are stored. Should be placed inside `/path/to/your/data/{set_name}`.
    - **img_name**: file name of the input image source. Should be stored in `/path/to/your/data/{set_name}/img/{img_name}`.
    - **label_name**: file name of the refering mask image (label/ground truth). Should be stored in `/path/to/your/data/{set_name}/label/{label_name}`

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
One should notice that, for training SAM-like architectures (e.g., SAMUS), it is expected that the total number of label masks per input image is equivalent to the number of individual objects to be segmented. In this scenario, the CSV file containing the metadata would have multiple rows with the same `img_name`, but with different `label_name`. For the training of semantic segmentation models, we would expect one `label_name` per `img_name`.

Some samples of datasets may be found at `data/`. In this ones, `ribs_da1` shows how the data is organized for training/evaluating semantic segmentation models, as `ribs_set_sep_sample` shows how it should be organized for the traning/evaluation of SAM-like architectures.

## [OPTIONAL] Docker Build
If it is of your interest to have the environement set up in a [Docker](https://www.docker.com/) environment, a base `Dockerfile` may be seen at `docker/Dockerfile`. In order to build it, please follow the steps:

```bash
# change your path to the docker folder
cd docker/
# build the docker image
docker build -t seg-lib-image:latest .
# verify if the image appears on the list of images
docker images
# run the docker image
docker run -it seg-lib-image:latest /bin/bash
```

It is also possible build a [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/index.html#) image using the `data/image.def` file. To do it so, execute the commands:
```bash
# build the image file
singularity build image.sif image.def
# run a script on the built image
singularity run \
    --bind /path/to/files:/mnt \
    --nv image.sif \
    python MYSCRIPT.py
```

It is worth mentioning that `sudo` access may be required for running some of the listed commands.

## License
This project is licensed under the **Apache License 2.0**, which allows you to freely use, modify, and distribute the code, even for commercial purposes. The license also provides protection by granting patent rights from contributors. However, there are a few responsibilities: you must include a copy of the license in any distribution, provide proper attribution to the original authors, and if you make any modifications to the code, a notice of those changes should be included. There is no requirement to release your modified code. For full details, refer to the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).


## Questions & Change Requests
If you have any questions regarding the projects or if you face any problems during the execution of our sources/scripts, please feel free to create an issue on this repository or to contact any of the authors.