import argparse
import cv2
import os

from tqdm import tqdm

from seg_lib.io.labelme import PointFormatter, Labelme2Yolov8
from seg_lib.io.files import read_json, dump_json
from seg_lib.io.image import img_b64_to_arr
     
def generate(
        input_files: list,
        formatter: PointFormatter = None,
        output_path: str = None,
        tag: str = None
    ):
    labels = []
    tag = '' if tag is None or len(tag) < 1 else f'{tag}_'

    if formatter is None:
        raise ValueError('Formatter should not be None!')

    data_path = os.path.join(output_path, formatter.mode, 'images')
    labels_path = os.path.join(output_path, formatter.mode, 'labels')
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for file_path in tqdm(input_files):
        # read the labelme json file
        data = read_json(file_path)
        fname = data['imagePath'].replace('\\', '/')
        fname = os.path.basename(fname)
        fname = os.path.splitext(fname)[0]
        fname = tag + fname

        # decode and save output
        img = img_b64_to_arr(data['imageData'])
        cv2.imwrite(os.path.join(data_path, f'{fname}.png'), img)
        del img

        w, h = data['imageWidth'], data['imageHeight']
        # process the points
        output_file = os.path.join(labels_path, f'{fname}.txt')
        fout = open(output_file, 'w')
        for shape in data['shapes']:
            if shape['label'] not in labels:
                labels.append(shape['label'])

            label_idx = labels.index(shape['label'])
            formatted_data = formatter(shape['points'], h, w)
            fout.write(f'{label_idx} ' + ' '.join(formatted_data) + '\n')
        fout.close()

    dump_json(
        os.path.join(output_path, formatter.mode, 'labels.txt'),
        {i: label for i, label in enumerate(labels)}
    )

def get_files(input_path: str):
    return [
        _file.path for _file in os.scandir(input_path)
        if _file.name.endswith('json')
    ]

def get_args():
    parser = argparse.ArgumentParser(
        prog='Labelme2YoloV8',
        description='Converts a Labelme output JSON to the YOLOv8 format.')

    parser.add_argument(
        '-i', '--input-path',
        type=str, help='Path to the input directory with the JSON files.')
    parser.add_argument(
        '-f', '--input-file',
        type=str, help='Path to a single JSON file.')
    parser.add_argument(
        '-o', '--output-path',
        type=str, default='yolov8_dataset/',
        help='Path to store the ouputs.')
    parser.add_argument(
        '-m', '--mode',
        default='bbox', choices={'bbox', 'seg', 'both'},
        type=str, help='Execution mode.')
    parser.add_argument(
        '-t', '--tag',
        type=str, help='Output filenames prefix tag.')
    
    args = parser.parse_args()

    if args.input_path is None and args.input_file is None:
        raise ValueError('Missing arguments: "--input-path" or "--input-file"')

    return args

if __name__ == '__main__':
    args = get_args()
    json_file_paths = (
        get_files(args.input_path) if args.input_path is not None
        else [args.input_file]
    )

    if args.mode in {'seg', 'both'}:
        print('Generating segmentation labels')
        generate(
            json_file_paths,
            formatter=Labelme2Yolov8.seg,
            output_path=args.output_path,
            tag=args.tag
        )
    if args.mode in {'bbox', 'both'}:
        print('Generating boxes labels')
        generate(
            json_file_paths,
            formatter=Labelme2Yolov8.box,
            output_path=args.output_path,
            tag=args.tag
        )
