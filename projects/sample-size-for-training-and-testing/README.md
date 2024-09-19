# Sample Size for Training and Testing: Segment Anything models and supervised approaches

Repository containing the source code used for training segmentation models (SAMUS, PVT-Seg and CAFE-NET) and applying late fusion on segmentation ensembles on the work: 
- `Sample Size for sTraining and Testing: Segment Anything models and supervised approaches` (lib version: [v1.0.3]()).

## Project Structure
This project is structured as follows:
- `notebooks`: contains the Jupyter notebooks used during experimentation;
- `samus`:
- `seg`: 
- `seg_n_sam`: 

## Dependency installation
In order to run the here presented source codes, it is required that the user has the `seg_lib` library installed on version `1.0.3`:

```bash
pip install seg_lib==1.0.3
```

More details on the installation process may be found on the README.md file contained to the root of this repository

## Data Input Format
The scripts expect that the data are described in a CSV file, contained in a `metadata` folder existing in `--data_path`. In order to set the data in the desired format, please follow the steps described as follows:

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

## Data Download
This work used the [VinDr-RibCXR](https://github.com/vinbigdata-medical/MIDL2021-VinDr-RibCXR) dataset. The version of the used images and masks may be found at [Google Drive](https://www.google.com/drive/). Different from the original version, the used dataset was used for semantic segmentation and therefore are presented as pairs of input and mask image files. In order to get the dataset as in the original input format, please recall to the original authors and create a request to them. 

## Running the Scripts
All of the scripts (or jobs) may be found in the `scripts` folder. It includes scripts for testing, training, fusion methods, etc. To run any of them, simply activate the conda environment that was previously set up and run the command as:

```bash
# run the scripts individually
python submodule/script.py [ARGS]
python submodule/script_2.py [ARGS]

# or run all of them sequentially
./submodule/run.sh
```

A brief description of each one of the scripts may be found in READMEs contained inside each submodule. Currently, three submodules are available:
- `samus`: scripts developed for training/evaluating SAMUS and its variations;

    - `train.py`: script using for training the SAMUS model adapation approach;
    - `eval.py`: evaluate the trained model using the standard dataloader;
    - `test.py`: evaluate the model using the Predictor class (as in the original SAM model).
- `seg`: scripts developed for the general training/evaluation of segmentation models;
    - `train.py`: train one of the available segmentation models architecture (SegPVT, CAFE)
- `seg_n_sam`: scripts developed for the segmentation + SAM fusion pipeline.

In order to re-achieve our work's results, please execute the scripts in the following order:

```bash
./samus/run.sh
./seg/run.sh
./seg_n_sam/run.sh
```

Finally, in order to perform experiments with late fusion from the segmentation models' output logits, please check the notebook `seg_fusion.ipynb`, located under `notebooks/`.

## References
```
@article{kirillov2023segany,
    title={Segment Anything},
    author={Kirillov, Alexander
        and Mintun, Eric
        and Ravi, Nikhila
        and Mao, Hanzi
        and Rolland, Chloe
        and Gustafson, Laura
        and Xiao, Tete
        and Whitehead, Spencer
        and Berg, Alexander C.
        and Lo, Wan-Yen
        and Doll{\'a}r, Piotr
        and Girshick, Ross},
    journal={arXiv:2304.02643},
    year={2023}
}

@misc{lin2023samus,
    title={SAMUS: Adapting Segment Anything Model for Clinically-Friendly and Generalizable Ultrasound Image Segmentation}, 
    author={Xian Lin
        and Yangyang Xiang
        and Li Zhang
        and Xin Yang
        and Zengqiang Yan
        and Li Yu},
    year={2023},
    eprint={2309.06824},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{nguyen2021,
      title={VinDr-RibCXR: A Benchmark Dataset for Automatic Segmentation and Labeling of Individual Ribs on Chest X-rays}, 
      author={Hoang C. Nguyen and Tung T. Le and Hieu H. Pham and Ha Q. Nguyen},
      year={2021},
      eprint={2107.01327},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2107.01327}, 
}
```