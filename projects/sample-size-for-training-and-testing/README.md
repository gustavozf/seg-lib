# Sample Size for Training and Testing: Segment Anything models and supervised approaches

Repository containing the source code used for obtaining the results described in the chapter `Sample Size for Training and Testing: Segment Anything models and supervised approaches`.

## Abstract
> The problem of determining the minimum amount of data required to train and test an artificial intelligence model has received substantial attention in the literature. In this chapter, we first review key concepts on the topic, then we survey selected theoretical and experimental results from the open literature, and in the end we present, as a case study, experiments we performed ourselves on the semantic segmentation of radiology images. A discussion from both a theoretical and an experimental point of view is required because the two approaches have complementary insights to offer. Theory provides general guidelines to avoid pitfalls during all phases of design: data collection, model design, training, and testing. Experimental results show what the current state of the art is in terms of performance and provide practical advice on which techniques have proven to be the most effective; for a more comprehensive study, we tested both supervised and zero-shot segmentation approaches, such as the “Segment Anything Model” (better known as SAM).

## Dependency installation
To run the here presented source codes, it is required that the user has the `seg-lib` library installed on version `1.0.4`:

```bash
pip install seg-lib==1.0.4
```

More details on the installation process may be found on the README.md file contained to the root of [seg-lib repository](https://github.com/gustavozf/seg-lib).

## Project Setup
To run the experiments, please download the required resources from [Google Drive](https://drive.google.com/file/d/1VW7Tp5YwNtFclg2zoCDkshJaLQ5-Ksba/view?usp=sharing) (password: `wdp,F)t/#M_k2v}Ly.G]V;`). The downloaded ZIP file will contain the following data folders:
- `ribs_full`: original VinDr-RibCXR dataset, but adapted for sematic segmentation tasks.
- `ribs_da1` and `ribs_da2`: same as `rubs_full`, but with two different offline data augmentation protocols applied;
- `ribs_small_da1` and `ribs_small_da2`: same as `ribs_da1` and `ribs_da1`, but with the small version of the ribs dataset;
- `ribs_set_sep`: original dataset, but with one mask for each rib set (left and right). Such dataset is used for training SAM-like architectures (e.g., SAMUS).

All of the CSV metadata files may be found in the `data/metadata/`. More details about the datasets may be found on our published chapter. Additionally, in order to get the VinDr-RibCXR dataset in its original input format, please recall to the [original authors' page](https://github.com/vinbigdata-medical/MIDL2021-VinDr-RibCXR) and file a request to them.

The ZIP file also included the model weights for training the segmentations models:
- `pvt_v2_b2.pth`: PVTv2 (B2) pretrained weights, used as a backbone for the SegPVT and CAFE-net semantic segmentation models;
- `sam_vit_b_01ec64.pth`: pretrained SAM weights (ViT-b encoder), used for training SAMUS. 

Extract the ZIP file to a local path and update the `BASE_PATH` variables on the `run.sh` scripts (more details on the next section).

## Running the Scripts
All of the scripts may be found split into three main folders:
- `samus`: scripts developed for training/evaluating SAMUS. Includes:

    - `train.py`: script used for training the SAMUS model;
    - `eval.py`: evaluate the trained model using the standard dataloader. Metrics are generated with the masks resized to the model's embedding size (similar to the validation step of the training script);
    - `test.py`: evaluate the model using the Predictor class (as in the original SAM model). Metrics are generated with the masks in the original input size.

- `seg`: scripts developed for the general training/evaluation of segmentation models (SegPVT or CAFE-Net). Includes:

    - `train.py`: train one of the available segmentation models architecture (SegPVT, CAFE);
    - `eval.py`: evaluate the trained segmentation models and generate the segmentation metrics.

- `fusion_seg_n_sam`: scripts developed for the segmentation + SAM fusion pipeline. Includes:

    - `inference.py`: use the segmentation model's evaluation outputs to sample input prompts that are later fed to SAM/SAMUS, aiming to get a new prediction on the dataset;
    - `eval.py`: generate metrics on the performance of the segmentation model (used for sampling the point prompts), SAM/SAMUS model (used for generating the new inferences) and of new predictions generated from the fusion of both of their logits.

To run any of them, simply activate the conda environment that was previously set up and run the command as:

```bash
# train, eval and test SAMUS model
cd samus/
python train.py [ARGS]
python eval.py [ARGS]
python test.py [ARGS]
# train and eval seg model
cd ../seg
python train.py [ARGS]
python eval.py [ARGS]
# infer and eval the fusion approach
cd ../fusion_seg_n_sam
python inference.py [ARGS]
python eval.py [ARGS]
```

To check the arguments of a target script, please run `python script --help` or check the `get_args()` function contained inside of it. One other alternative would be to run the `run.sh` files contained inside each subfolder. Such as:


```bash
# train, eval and test SAMUS model
cd samus/
./run.sh
# train and eval seg model
cd ../seg
./run.sh
# infer and eval the fusion approach
cd ../fusion_seg_n_sam
./run.sh
```

These ones already have default values for each argument. Before executing any `run.sh` file, please overwrite at least the `BASE_PATH` variable. This one is assumed to be the path location of the extracted ZIP file, containing the `data/` and `pretrained_models/` folders. Also, overwrite the `DATASET_DESCRIPTOR` variable accordingly to the target dataset.

Finally, in order to perform experiments with ensembles of segmentation models (using their output logits), please check the notebook `seg_fusion.ipynb`, located under `notebooks/`.

## Project Citation
If you want to cite our chapter or the source codes contained in this repository, please used the citation (bibtex format):

```
@Inbook{Cuza2024,
  author="Cuza, Daniela
    and Fantozzi, Carlo
    and Nanni, Loris
    and Fusaro, Daniel
    and Felipe, Gustavo Zanoni
    and Brahnam, Sheryl",
  editor="Lim, Chee-Peng
    and Vaidya, Ashlesha
    and Jain, Nikhil
    and Favorskaya, Margarita N.
    and Jain, Lakhmi C.",
  title="Sample Size for Training and Testing: Segment Anything Models and Supervised Approaches",
  bookTitle="Advances in Intelligent Healthcare Delivery and Management: Research Papers in Honour of Professor Maria Virvou for Invaluable Contributions",
  year="2024",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="107--145",
  isbn="978-3-031-65430-5",
  doi="10.1007/978-3-031-65430-5_6",
  url="https://doi.org/10.1007/978-3-031-65430-5_6"
}
```