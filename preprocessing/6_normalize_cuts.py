import cv2
import mediapipe
import numpy as np
import pandas as pd
import glob
import os
import tqdm
import matplotlib.pyplot as plt

IMG_SIZE=(100,100)
GRAY = False

sizes = []
for img_id in os.listdir('cut_faces'):
    img = cv2.imread('cut_faces/' + img_id)
    if GRAY:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)

    w, h = img.shape[0:2]
    max_dim = max(w, h)
    pad_w = (max_dim - w) // 2
    pad_h = (max_dim - h) // 2
    padded_img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    norm_img = cv2.resize(padded_img, IMG_SIZE)
    
    FOLDER = 'norm_cut_faces/' if GRAY else 'color_norm_cut_faces/'
    cv2.imwrite(FOLDER + img_id, norm_img)
