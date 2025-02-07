"""
MIT License

Copyright (c) 2023 Mehmet OKUYAR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://github.com/MehmetOKUYAR/Convert-Mask2Yolo/blob/main/converter_codes.py
"""

import os

import cv2

INPUT_PATH = '/path/to/masks'
OUTPUT_PATH = '/path/to/outputs'
CLASS_ID = 0

img_paths = [
    os.path.join(root, _file)
    for root, _, _files in os.walk(INPUT_PATH)
    for _file in _files
    if _file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))
]

# for each image in the mask path
count = 0
for img_path in img_paths:
    count +=1

    # -------- read image and get width and height ----------------
    img = cv2.imread(img_path)
    width = img.shape[1]
    height = img.shape[0]

    kopya = img.copy()
    kopya = cv2.cvtColor(kopya, cv2.COLOR_RGB2GRAY)

    # -------- get contours ----------------
    blur = cv2.cv2.GaussianBlur(kopya,(5,5),0)
    thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)[1]
    kontur_1 = cv2.findContours(
        thresh.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(kontur_1[0]) == 1:

        kontur = kontur_1[0:-1][0][0]

        x_list = [[]]
        y_list = [[]]
        for num, i in enumerate(kontur):
            x_list[0].append(kontur[num][0][0])
            y_list[0].append(kontur[num][0][1])

    elif len(kontur_1[0]) > 1:

        kontur = kontur_1[0:-1][0]
        x_list = []
        y_list = []

        for i in range(len(kontur)):
            kontur_2 = kontur[i]
            xara_list = []
            yara_list = []

            for num, i in enumerate(kontur_2):
                xara_list.append(kontur_2[num][0][0])
                yara_list.append(kontur_2[num][0][1])

            x_list.append(xara_list)
            y_list.append(yara_list)
    
    # -------- get image name ----------------
    name = img_path.replace("\\","/").split("/")[-1]
    name, extantion = os.path.splitext(name)

    # -------- write coordinates in txt file ----------------
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    fname = os.path.join(OUTPUT_PATH, f"{name}.txt")
    f = open(fname, "w")
    for i in range(len(x_list)):
        f.write(str(CLASS_ID)) # class id
        for j in range(len(x_list[i])):
            # coordinates
            x = round(x_list[i][j]/width, 4)
            y = round(y_list[i][j]/height,4)
            f.write(f" {x} {y}")
            
        f.write("\n")   
    f.close()