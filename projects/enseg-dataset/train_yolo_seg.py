import gc
import os
import tempfile

from ultralytics import YOLO

from seg_lib.models.device import CLEAN_CACHE, get_device
import argparse

ANIMAL_TAGS = ['2C', '4C', '5C', '22TW', '23TW', '28TW']
MODEL_NAMES = [
    'yolov8n',
    'yolov8s',
    'yolov8m',
    'yolov8l',
    'yolov8x',
    'yolov9c',
    'yolov9e',
    'yolo11n',
    'yolo11s',
    'yolo11m',
    'yolo11l',
    'yolo11x'
]

def get_config_txt(
        data_path: str,
        train_ids: list[str],
        val_id: str
    ) -> str:
    """ This function generates the configuration file for 
        training the YOLO model. It is used to train the model
        with the ultralytics library.

        This is just an example of how the configuration file
        should look like. It is not the actual configuration
        file used in the project.
    """
    train_paths = [
        f'{_id}/seg/images'
        for _id in train_ids
    ]
    val_path = f'{val_id}/seg/images'
    return (
        f"path: '{data_path}'\n"
        f"train: {train_paths}\n"
        f"val: '{val_path}'\n"
        "\n"
        "names:\n"
        "  0: 'enc'\n"
    )

def train_model(
        train_tags: list[str],
        val_tag: str,
        model_name: str = 'yolov8n',
        data_path: str = 'data/',
        output_path: str = 'outputs/',
        device: str = 'cpu',
        train_batch_size: int = -1,
        val_batch_size: int = 2,
        n_epochs: int = 100,
        image_size: int = 640
    ) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, 'yolo.yaml')
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(get_config_txt(data_path, train_tags, val_tag))

        model = YOLO(f'{model_name}-seg.pt')
        # train procedure from ultralytics.
        # data augmentations are applied automatically. Such as:
        # - Blur(p=0.01, blur_limit=(3, 7))
        # - MedianBlur(p=0.01, blur_limit=(3, 7))
        # - ToGray(p=0.01, num_output_channels=3, method='weighted_average')
        # - CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
        # - etc.
        model.train(
            project=output_path,
            name=os.path.join(val_tag, 'train'),
            data=temp_file_path,
            optimizer='AdamW',
            lr0=0.002,
            momentum=0.9,
            epochs=n_epochs,
            imgsz=image_size,
            batch=train_batch_size,
            device=device)
        model.val(
            project=output_path,
            name=os.path.join(val_tag, 'val'),
            data=temp_file_path,
            save_json=True,
            plots=True,
            batch=val_batch_size,
            device=device)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train YOLO segmentation model.'
    )
    parser.add_argument(
        '--data_path',
        type=str, required=True,
        help='Path to the data files.')
    parser.add_argument(
        '--output_path',
        type=str, default='outputs/',
        help='Path to save the outputs.')
    parser.add_argument(
        '--train_batch_size',
        type=int, default=-1,
        help='Batch size for training.')
    parser.add_argument(
        '--val_batch_size',
        type=int, default=2,
        help='Batch size for validation.')
    parser.add_argument(
        '--epochs',
        type=int, default=100,
        help='Number of epochs for training.')
    parser.add_argument(
        '--image_size',
        type=int, default=640,
        help='Image size for training.')

    return parser.parse_args()

def main():
    args = parse_args()
    print('Preparing the data...')
    device: str = get_device()
    print(args.data_path)

    print('Running experiments...')
    for model_name in MODEL_NAMES:
        print('Training model: ', model_name)
        for val_tag in ANIMAL_TAGS:
            curr_out_path = os.path.join(args.output_path, model_name)

            if os.path.exists(
                os.path.join(curr_out_path, val_tag, 'val', 'predictions.json')
            ):
                continue

            train_tags = [tag for tag in ANIMAL_TAGS if tag != val_tag]
            train_model(
                train_tags, val_tag,
                model_name=model_name,
                data_path=args.data_path,
                output_path=curr_out_path,
                train_batch_size=args.train_batch_size,
                val_batch_size=args.val_batch_size,
                n_epochs=args.epochs,
                image_size=args.image_size,
                device=device
            )
            gc.collect()
            CLEAN_CACHE[device]()
    
if __name__ == '__main__':
    main()