"""This is a file for training a model by showing images from a folder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import random
import numpy as np
import PIL.Image
import time


from pymuse.inputstream.muse_constants import MUSE_EEG_ACQUISITION_FREQUENCY
from pymuse.pipelinestages.pipeline_stage import PipelineStage
from pymuse.signal import SignalData
from pymuse.pipeline import Pipeline
from pipeline.muse_osc_input_stream import MuseOSCInputStream
from pymuse.inputstream.muse_constants import MUSE_ACQUISITION_FREQUENCIES, MUSE_OSC_PATH
from pymuse.configureshutdown import configure_shutdown
from pymuse.inputstream.constants import DEFAULT_UDP_PORT, LOCALHOST, SIGNAL_QUEUE_LENGTH
from pymuse.signal import Signal, SignalData
from bci_client import MuseClient
from utils import save_img_pair


BASE_IMG_DIR = "/Users/petervantoth/brain_ai_painting_project/wikiart/wikiart"
IMG_DIRS = os.listdir(BASE_IMG_DIR)
TRAIN_DIR = "/Users/petervantoth/brain_ai_painting_project/train_data_pix2pix"


def pick_image_path_random():
    """Picks an image randomly."""
    rnd_dir = random.choice(IMG_DIRS)
    img_files = os.listdir(BASE_IMG_DIR + "/" + rnd_dir)
    rnd_img = random.choice(img_files)

    return BASE_IMG_DIR + "/" + rnd_dir + "/" + rnd_img


if __name__ == "__main__":
    client = MuseClient()
    channels = ['eeg']

    muse_osc_input_stream = MuseOSCInputStream(channels, ip="192.168.1.10", port=6500)

    pipeline = Pipeline(muse_osc_input_stream.get_signal(channels[0]),
                        client)

    configure_shutdown(muse_osc_input_stream, pipeline)
    pipeline.start()
    muse_osc_input_stream.start()
    print("start")

    img = cv2.imread(pick_image_path_random())
    img = cv2.resize(img, (256, 256))
    cv2.imshow("image", img)
    cv2.waitKey(0)

    num_image_pairs = 0

    while True:
        brain_data = client.output()
        if brain_data is not None:

            img_arr = np.array(img)
            # print("image data {}".format(img_arr.shape))
            if img_arr.shape[0] != 256 or img_arr.shape[1] != 256:
                print(img_arr.shape)
            seconds_since_epoch = time.strftime("%Y%m%d-%H%M%S")
            save_img_pair(brain_data, img, TRAIN_DIR + "/" + str(seconds_since_epoch) + ".jpg")
            num_image_pairs += 1
            print("Num images saved {}".format(num_image_pairs))

            img = cv2.imread(pick_image_path_random())
            img = cv2.resize(img, (256, 256))
            cv2.imshow("image", img)
            cv2.waitKey(0)


