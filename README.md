# SegLib

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A PyTorch library for building, training, and evaluating image segmentation models. Developed jointly by researchers at the [Università degli Studi di Padova](https://www.unipd.it/) and the [Universidade Estadual de Maringá](https://www.uem.br/).

[SegLib](https://github.com/gustavozf/seg-lib) ships semantic and promptable segmenters behind a shared interface, together with trainers, dataloaders, losses, prompt sampling, and evaluation utilities. Paper-specific experiments live under [`projects/`](projects/).

## Contents

- [Features](#features)
- [Repository layout](#repository-layout)
- [Research projects](#research-projects)
- [Installation](#installation)
- [Data format](#data-format)
- [Container images](#container-images)
- [License](#license)
- [Questions and change requests](#questions-and-change-requests)

## Features

Semantic models are constructed with `seg_selector`. Promptable models are constructed with `sam_selector` and wrapped for inference by `predictor_selector`. Both selectors live in `seg_lib/models/selector.py`.

### Models

| Model | Task | Selector name | Notes |
| --- | --- | --- | --- |
| SegPVT | Semantic | `SegPVT` | PolyPVT with a PVTv2 backbone |
| CAFE-Net | Semantic | `CAFE` | PVTv2 backbone |
| SAM | Promptable | `SAM` | ViT-B, ViT-L, or ViT-H. Input size 1024 |
| SAM 2 | Promptable | `SAMv2` | Hiera tiny, small, base+, or large |
| SAM 2.1 | Promptable | `SAMv2.1` | Same Hiera sizes as SAM 2 |
| SAM-Med2D | Promptable | `SAM-Med2D` | Input size 256, with an encoder adapter |
| SAMUS | Promptable | `SAMUS` | Ultrasound-oriented SAM. Input size 256 |

Each promptable model has a matching predictor: `SamPredictor`, `Sam2Predictor`, `SamMed2DPredictor`, or `SamusPredictor`. `Sam2Predictor` serves both SAM 2 and SAM 2.1. A Cellpose-SAM implementation is also included under `seg_lib/models/cpsam`. SAM 2's image model is vendored under `seg_lib/models/sam2/`, so the upstream package is not a dependency. Weights are still downloaded separately.

### Trainers

| Trainer | Used for | Behavior |
| --- | --- | --- |
| `SegmentationModelTrainer` | Single-output semantic models | Train and validation loops, checkpointing, and mask metrics |
| `PVTv2ModelTrainer` | SegPVT and CAFE-Net | Sums the loss across auxiliary decoder heads |
| `SamusModelTrainer` | SAMUS | Accepts point and box prompts and computes the loss on low-resolution logits |

`PVTv2ModelTrainer` and `SamusModelTrainer` both extend `SegmentationModelTrainer`.

### Training stack

- **Losses.** Dice, focal, and Dice combined with binary cross-entropy. `select_loss_by_model` picks a combination for the target architecture.
- **Data.** `SegGeneralDataset` for semantic segmentation and `SamGeneralDataset` for promptable models, with weak, standard, and SAMUS-specific augmentation.
- **Prompts.** Training and inference samplers for points and boxes (`Sampler`, `TrainPromptSampler`, `MaskInferenceAfterSample`).
- **Evaluation.** Dice, IoU, MAE, weighted F-measure, and E-measure, plus logit fusion between a specialist segmenter and a SAM-family model (`LogitsFusion`, `KittlerLogitsFusion`).
- **I/O.** Image reading and LabelMe-to-YOLOv8 conversion.

## Repository layout

| Path | Contents |
| --- | --- |
| [`seg_lib/`](seg_lib/) | Shared library: models, trainers, dataloaders, losses, prompts, and evaluation |
| [`projects/`](projects/) | Scripts and notebooks for the published studies |
| [`data/`](data/) | Sample metadata in the expected training format |
| [`docker/`](docker/) | Dockerfile and Singularity definition |
| [`tools/`](tools/) | Standalone converters and utilities |

The package is built with Poetry. Dependencies are declared in [`pyproject.toml`](pyproject.toml).

## Research projects

Each study has its own README and BibTeX citation inside its folder.

### ENSeg dataset

Scripts for the segmentation of enteric neuron cells on microscopy images.

- **Title:** [ENSeg: A Novel Dataset and Method for the Segmentation of Enteric Neuron Cells on Microscopy Images](https://www.mdpi.com/2076-3417/15/3/1046)
- **Authors:** Gustavo Zanoni Felipe, Loris Nanni, Isadora Goulart Garcia, Jacqueline Nelisis Zanoni, and Yandre Maldonado e Gomes da Costa
- **Venue:** Applied Sciences (MDPI), 2025
- **Code:** [`projects/enseg-dataset/`](projects/enseg-dataset/)
- **Citation:** [`projects/enseg-dataset/CITATION.bib`](projects/enseg-dataset/citation.bib)

### Sample size for training and testing

Scripts for a study of how much labeled data semantic segmenters and Segment Anything models need, using radiology images.

- **Title:** [Sample Size for Training and Testing: Segment Anything Models and Supervised Approaches](https://doi.org/10.1007/978-3-031-65430-5_6)
- **Authors:** Daniela Cuza, Carlo Fantozzi, Loris Nanni, Daniel Fusaro, Gustavo Zanoni Felipe, and Sheryl Brahnam
- **Venue:** Advances in Intelligent Healthcare Delivery and Management (Springer), 2024
- **Code:** [`projects/sample-size-for-training-and-testing/`](projects/sample-size-for-training-and-testing/)
- **Citation:** [`projects/sample-size-for-training-and-testing/CITATION.bib`](projects/sample-size-for-training-and-testing/CITATION.bib)

### Improving existing segmentators

Notebooks that extend the zero-shot fusion method used as the base for the sample-size study. The original implementation is in the [upstream repository](https://github.com/LorisNanni/Improving-existing-segmentators-performance-with-zero-shot-segmentators).

- **Title:** [Improving Existing Segmentators Performance with Zero-Shot Segmentators](https://doi.org/10.3390/e25111502)
- **Authors:** Loris Nanni, Daniel Fusaro, Carlo Fantozzi, and Alberto Pretto
- **Venue:** Entropy (MDPI), 2023
- **Code:** [`projects/improving-existing-segmentators-performance/`](projects/improving-existing-segmentators-performance/)
- **Citation:** [`projects/improving-existing-segmentators-performance/CITATION.bib`](projects/improving-existing-segmentators-performance/CITATION.bib)

## Installation

SegLib requires Python 3.10. A dedicated conda environment keeps the PyTorch stack isolated:

```bash
conda create -n seg_lib_env python==3.10
conda activate seg_lib_env
```

### From PyPI

```bash
pip install seg-lib
# or a specific release
pip install seg-lib==1.0.5
```

### From source

```bash
pip install poetry
poetry build
pip install dist/seg_lib-1.0.5-py3-none-any.whl
```

A pre-built wheel is also available in `dist/`.

## Data format

Dataloaders read a CSV descriptor from a `metadata` folder inside the dataset root.

1. Create `/path/to/your/data/metadata`.
2. Add `{dataset_name}.csv`.
3. Include at least these columns:

| Column | Meaning |
| --- | --- |
| `split` | `train`, `val`, or `test` |
| `class_id` | Integer class index. Use `1` for binary masks |
| `subset` | Folder under the dataset root that holds this sample |
| `img_name` | Image file in `{subset}/img/` |
| `label_name` | Mask file in `{subset}/label/` |

Example rows:

```text
   split  class_id         subset                    img_name                     label_name
0  train         1  ribs_original  VinDr_RibCXR_train_000.png  VinDr_RibCXR_train_000_R1.bmp
1  train         1  ribs_original  VinDr_RibCXR_train_000.png  VinDr_RibCXR_train_000_R2.bmp
2  train         1  ribs_original  VinDr_RibCXR_train_000.png  VinDr_RibCXR_train_000_R3.bmp
```

Expected directory layout:

```text
metadata/
  dataset_descriptor.csv
subset_1/
  img/
    img_name_1.png
  label/
    img_name_1.bmp
subset_n/
  img/
  label/
```

Semantic models expect one mask per image. SAM-like models expect one mask per object, so the same `img_name` can appear on several rows with different `label_name` values.

Sample descriptors are in [`data/metadata/`](data/metadata/). `ribs_da1.csv` follows the semantic layout. `dataset_set_sep_sample.csv` follows the SAM-like layout.

## Container images

### Docker

[`docker/Dockerfile`](docker/Dockerfile) starts from the PyTorch 2.1.2 CUDA 11.8 image and installs SegLib:

```bash
cd docker/
docker build -t seg-lib-image:latest .
docker run -it seg-lib-image:latest /bin/bash
```

### Singularity

Build from [`docker/image.def`](docker/image.def):

```bash
singularity build image.sif image.def
singularity run \
    --bind /path/to/files:/mnt \
    --nv image.sif \
    python MYSCRIPT.py
```

Some of these commands require `sudo`.

## License

SegLib is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). You may use, modify, and distribute the code, including for commercial purposes. Distribute a copy of the license, keep attribution to the original authors, and note any modifications. Modified versions do not have to be released.

## Questions and change requests

Open an issue on this repository, or contact one of the authors, if a script fails or a project needs clarification.
