import os
import glob
import json
import logging

def read_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as _f:
        return json.load(_f)
    
def dump_json(file_path: str, data: dict):
    with open(file_path, 'w', encoding='utf8') as _f:
        json.dump(data, _f, ensure_ascii=False, indent=2)
    

def get_sorted_paths(paths):
    bn = [os.path.basename(path[:-4]).zfill(6) for path in paths]
    return [
        paths[i]
        for i in sorted(range(len(bn)), key=lambda k: bn[k])
    ]

def __get_data_paths(dataset_path, segmentator_name):
    if not os.path.isdir(dataset_path):
        logging.error("Provided dataset does not exist!")
        return [[], [], [], []]
    
    orig_images_folder = os.path.join(dataset_path, "imgs")
    gt_folder          = os.path.join(dataset_path, "gt")
    segmentator_folder = os.path.join(
        dataset_path, "segmentator_" + segmentator_name
    )
    
    ## Load input images ##
    test_imgs = get_sorted_paths(
        glob.glob(os.path.join(orig_images_folder, '*'))
    )    
    ## Load GT masks ##
    gt_masks = get_sorted_paths(glob.glob(os.path.join(gt_folder, '*')))
    ## Load DeepLabV3+ produced binary masks ##
    segmentator_bmasks = get_sorted_paths(
        glob.glob(os.path.join(segmentator_folder, '*.bmp'))
    )
    ## Load DeepLabV3+ produced 3D masks ##
    segmentator_rmasks =  get_sorted_paths(
        glob.glob(os.path.join(segmentator_folder, '*.png'))
    )
        
    return [test_imgs, gt_masks, segmentator_bmasks, segmentator_rmasks]

def load_data_paths(dataset_path: str, source_mask: str):
     paths = __get_data_paths(dataset_path, source_mask)
     lens = list(map(lambda i: len(i), paths))

     # if there are no 3D liogits, repeat the 2D ones for consistency purposes
     if lens[3] == 0:
          paths[3] = paths[2]
          lens[3] = lens[2]

     # check if the images and the ground truths have the same quantity
     img_has_gt_len = lens[0] == lens[1]
     # check if either of the masks (2D or 3D logits) matches with the len of
     # the images
     mask_has_img_len = (
        (lens[0] == lens[2] == lens[3])
          or (lens[0] == lens[2])
          or (lens[0] == lens[3])
     )
     if not (img_has_gt_len and mask_has_img_len):
        raise ValueError(f'Unbalanced datasets! {lens}')
     del lens

     return list(zip(*paths))