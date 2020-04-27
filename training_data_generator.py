"""This is a file for training a model by showing images from a folder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import random

IMG_DIR = "/Users/petervantoth/brain_ai_painting_project/wikiart/wikiart"

dirs = os.listdir(IMG_DIR)
num_files = 0
for dir in dirs:
    if dir != ".DS_Store":
        files = os.listdir(IMG_DIR + "/" + dir)
        num_files += len(files)

print(num_files)

# while True:
#     img_path = random.choice(imgs)
#     print(img_path)
#     img = cv2.imread(IMG_DIR + "/" + img_path)
#     print(img)
#     cv2.imshow("image", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()