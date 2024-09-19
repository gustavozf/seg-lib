import argparse
import os

import torch
from torch.utils.data import DataLoader

from seg_lib.dataloaders.sam_dataset import SamGeneralDataset
from seg_lib.io.files import read_json
from seg_lib.losses.combined_losses import MaskDiceAndBCELoss
from seg_lib.models.samus.build_sam_us import samus_model_registry
from seg_lib.prompt.train_sampler import TrainPromptSampler
from seg_lib.trainers import SamusModelTrainer

# total number of parallel workers used in the dataloader
N_CPUS = os.cpu_count()
N_GPUS = torch.cuda.device_count()
DEVICE = 'cuda' if N_GPUS > 0 else 'cpu'

def get_args():
    parser = argparse.ArgumentParser(
        prog='SamusModelEval',
        description='Evaluate a SAMUS model.'
    )

    parser.add_argument(
        '-i', '--input_path',
        required=True, type=str,
        help='Path to the training output path.')
    parser.add_argument(
        '-w', '--backbone_weights_path',
        required=False, type=str,
        help='Path to the backbone weights.')
    parser.add_argument(
        '-d', '--data_path',
        required=True, type=str,
        help='Path to the test data.')
    parser.add_argument(
        '-dd', '--data_desc_path',
        required=True, type=str,
        help=('Path to the data description file (CSV format). '
              'Read from `data_path`.'))
    parser.add_argument(
        '-b', '--batch_size',
        required=False, type=int, default=8,
        help='Batch size used for training and validation.')
    parser.add_argument(
        '-s', '--split_name',
        required=False, type=str, default='val',
        help=(
            'Data split name, used to filter the data samples on the '
            'metadata file.'
        )
    )
    
    return parser.parse_args()

def get_model(
        checkpoint_path: str,
        backbone_path: str,
        input_size: int = 256):
    model = samus_model_registry['vit_b'](
        encoder_input_size=input_size, checkpoint=backbone_path
    )
    model.to(torch.device(DEVICE))
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device(DEVICE)
    )
    model.load_state_dict(checkpoint)

    return model

def get_data(
        data_path: str, df_file_path: str,
        input_size: int = 256, embedding_size: int = 128,
        prompt_type: str = 'click', max_num_prompts: int = 1,
        batch_size: int = 8, split_name: str = 'val'):
    # prompt sampler used in order to get points from the image masks
    sampler = TrainPromptSampler(
        min_blob_count=10, max_num_prompts=max_num_prompts
    )

    # return image, mask, and filename
    val_dataset = SamGeneralDataset(
        data_path,
        split=split_name,  point_sampler=sampler,
        df_file_path=df_file_path,
        img_size=input_size,
        embedding_size=embedding_size,
        prompt=prompt_type,
        read_img_as_grayscale=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * max(1, N_GPUS),
        shuffle=False,
        num_workers=N_CPUS,
        pin_memory=True)
    
    return val_loader

def get_loss():
    pos_weight = torch.ones([1]) * 2
    pos_weight = pos_weight.to(device=torch.device(DEVICE))
    return MaskDiceAndBCELoss(pos_weight=pos_weight)

def main():
    eval_config = get_args()
    train_config = read_json(
        os.path.join(eval_config.input_path, 'config.json')
    )
    checkpoint_path = os.path.join(
        train_config['output_path'], f"best_{train_config['model_type']}.pth"
    )

    model = get_model(
        checkpoint_path,
        eval_config.backbone_weights_path,
        input_size=train_config['input_size'])
    val_ds = get_data(
        eval_config.data_path, eval_config.data_desc_path,
        input_size=train_config['input_size'],
        embedding_size=train_config['embedding_size'],
        prompt_type=train_config['prompt_type'],
        max_num_prompts=train_config['n_clicks'],
        batch_size=eval_config.batch_size,
        split_name=eval_config.split_name)
    loss_f = get_loss()

    trainer = SamusModelTrainer(model, None, loss_f, device=DEVICE)
    metrics = trainer.val_loop(val_ds)
    print(metrics)

if __name__ == '__main__':
    main()