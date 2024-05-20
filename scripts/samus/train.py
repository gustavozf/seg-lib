import argparse
import os
import random
from importlib.metadata import version

import torch
import numpy as np
from torch.utils.data import DataLoader

from seg_lib.dataloaders.data_aug import WeakImgSegAugmenter
from seg_lib.dataloaders.sam_dataset import SamGeneralDataset
from seg_lib.io.files import dump_json
from seg_lib.losses.combined_losses import MaskDiceAndBCELoss
from seg_lib.models.samus.build_sam_us import samus_model_registry
from seg_lib.prompt.train_sampler import TrainPromptSampler
from seg_lib.trainers import SamusModelTrainer

CONFIG = None
RANDOM_SEED = 1234
# total number of parallel workers used in the dataloader
N_GPUS = torch.cuda.device_count()
DEVICE = 'cuda' if N_GPUS > 0 else 'cpu'

def get_args():
    parser = argparse.ArgumentParser(
        prog='SegmentationModelTrainer',
        description='Train a segmentation model.'
    )

    parser.add_argument(
        '-t', '--model_type',
        required=False, type=str, default='SAMUS',
        help='Type of segmentation model architecture.')
    parser.add_argument(
        '-d', '--data_path',
        required=True, type=str,
        help='Path to the training data.')
    parser.add_argument(
        '-o', '--output_path',
        required=True, type=str,
        help='Path for saving the model outputs.')
    parser.add_argument(
        '-dd', '--data_desc_path',
        required=True, type=str,
        help=('Path to the data description file (CSV format). '
              'Read from `data_path`.'))
    parser.add_argument(
        '-c', '--checkpoint',
        required=False, type=str,
        help='Path to the pretrained base model checkpoint.')
    parser.add_argument(
        '-i', '--input_size',
        required=False, type=int, default=256,
        help='Image input size.')
    parser.add_argument(
        '-es', '--embedding_size',
        required=False, type=int, default=128,
        help='Embedding size. Used for SAM-like training procedure.')
    parser.add_argument(
        '-lr', '--learning_rate',
        required=False, type=float, default=0.0001,
        help='Base learning rate.')
    parser.add_argument(
        '-lrw', '--lr_warmup_period',
        required=False, type=int, default=0,
        help='Learning rate warmup period (0 = deactivated).')
    parser.add_argument(
        '-b', '--batch_size',
        required=False, type=int, default=8,
        help='Batch size used for training and validation.')
    parser.add_argument(
        '-e', '--epochs',
        required=False, type=int, default=200,
        help='total number of training epochs.')
    parser.add_argument(
        '-ef', '--eval_freq',
        required=False, type=int, default=1,
        help='The frequency of evaluate the model (in epochs).')
    parser.add_argument(
        '-sf', '--save_freq',
        required=False, type=int, default=1,
        help='The model dump frequency (in epochs).')
    parser.add_argument(
        '-p', '--prompt_type',
        required=False, type=str, default='bbox',
        choices={'bbox', 'click', 'all'},
        help='Input prompt type.')
    parser.add_argument(
        '-nc', '--n_clicks',
        required=False, type=int, default=1,
        help='Number of input point prompts (only when `--prompt_type=click`')
    parser.add_argument(
        '-n', '--n_cpus',
        required=False, type=int, default=os.cpu_count(),
        help='Total number of parallel workers used in the dataloader.')
    
    return parser.parse_args()

def set_seed():
    np.random.seed(RANDOM_SEED) # set random seed for numpy
    random.seed(RANDOM_SEED) # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED) # avoid hash random
    torch.manual_seed(RANDOM_SEED) # set random seed for CPU
    torch.cuda.manual_seed(RANDOM_SEED) # set random seed for one GPU
    torch.cuda.manual_seed_all(RANDOM_SEED) # set random seed for all GPU
    torch.backends.cudnn.deterministic = True # set random seed for convolution

def get_model():
    # load the base model using SAM checkpoint
    model = samus_model_registry['vit_b'](
        encoder_input_size=CONFIG.input_size,
        checkpoint=CONFIG.checkpoint)
    model.to(torch.device(DEVICE))
    
    # print the total number of parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f'Total_params: {pytorch_total_params}')
    del pytorch_total_params

    # if it is a multi-gpu approach, use data parallelism
    if N_GPUS > 1:
        return torch.nn.DataParallel(model)

    return model

def get_data():
    # prompt sampler used in order to get points from the image masks
    sampler = TrainPromptSampler(
        min_blob_count=10, max_num_prompts=CONFIG.n_clicks
    )

    # return image, mask, and filename
    loader_params = {
        'df_file_path': CONFIG.data_desc_path,
        'img_size': CONFIG.input_size,
        'embedding_size': CONFIG.embedding_size,
        'prompt': CONFIG.prompt_type,
        'read_img_as_grayscale': True
    }
    augmenter = WeakImgSegAugmenter(img_size=CONFIG.input_size)
    train_dataset = SamGeneralDataset(
        CONFIG.data_path, split='train', point_sampler=sampler,
        augmenter=augmenter, **loader_params)
    val_dataset = SamGeneralDataset(
        CONFIG.data_path, split='val',  point_sampler=sampler,
        **loader_params)

    batch_size = CONFIG.batch_size * max(1, N_GPUS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CONFIG.n_cpus,
        pin_memory=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CONFIG.n_cpus,
        pin_memory=True)
    
    return train_loader, val_loader

def get_optm(model: torch.nn.Module):
    if CONFIG.lr_warmup_period > 0:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=CONFIG.learning_rate / CONFIG.lr_warmup_period,
            betas=(0.9, 0.999),
            weight_decay=0.1)
    
    return torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False)


def main():
    global CONFIG
    CONFIG = get_args()

    # add the seed to make sure the results are reproducible
    set_seed()

    # get the data, init the model and get the optm+loss classes
    model = get_model()
    train_ds, val_ds = get_data()
    optimizer = get_optm(model)

    # select the loss
    pos_weight = torch.ones([1]) * 2
    pos_weight = pos_weight.cuda(device=torch.device(DEVICE))
    loss_f = MaskDiceAndBCELoss(pos_weight=pos_weight)

    # save the configuration to file
    os.makedirs(CONFIG.output_path, exist_ok=True)
    dump_json(
        os.path.join(CONFIG.output_path, 'config.json'),
        {
            **vars(CONFIG),
            'lib_version': version('unipd_sam_med'),
            'n_gpus': N_GPUS,
            'device': DEVICE,
            'seed': RANDOM_SEED
        }
    )

    trainer = SamusModelTrainer(model, optimizer, loss_f, device=DEVICE)
    history = trainer.fit(
        train_ds, val_ds,
        n_epochs=CONFIG.epochs,
        eval_freq=CONFIG.eval_freq,
        save_freq=CONFIG.save_freq,
        output_path=CONFIG.output_path,
        model_name=CONFIG.model_type
    )

if __name__ == '__main__':
    main()