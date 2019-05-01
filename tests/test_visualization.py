import json
import numpy as np
import os
import pickle

from PIL import Image
from vfn.utils.visualization import plot_bbox

SLIDING_WINDOWS = 'sliding windows'
GROUNDTRUTH = 'groundtruth'
PREDICT = 'predict'


def test_plot_bbox_sliding_windows():
    with open('../Flickr_21K/flickr_pro.pkl', 'rb') as f:
        db = pickle.load(f)

    filename = os.path.join('../Flickr_21K', os.path.basename(db[0]['url']))
    bboxes = [[(db[i]['crop'][0], db[i]['crop'][1]), (db[i]['crop'][2], db[i]['crop'][3])] for i in range(14)]

    image = Image.open(filename).convert('RGB')
    image = plot_bbox(image, bboxes, SLIDING_WINDOWS)
    image.save('test_plot_bbox_sliding_windows.jpg', 'JPEG')


def test_plot_bbox_groundtruth_predict():
    with open('../FCDB/FCDB-all.json', 'r') as f:
        db = json.load(f)

    filename = os.path.join('../FCDB', os.path.basename(db[0]['url']))
    bboxes = [[db[0]['crop'][0], db[0]['crop'][1], db[0]['crop'][2], db[0]['crop'][3]]]

    image = Image.open(filename).convert('RGB')
    image = plot_bbox(image, bboxes, GROUNDTRUTH)
    bboxes = [[50, 50, 512, 512]]
    image = plot_bbox(image, bboxes, PREDICT)
    image.save('test_plot_bbox_groundtruth_predict.jpg', 'JPEG')
