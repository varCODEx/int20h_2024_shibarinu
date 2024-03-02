import cv2
import mediapipe
import numpy as np
import pandas as pd
import glob
import os
import tqdm
import matplotlib.pyplot as plt

sizes = []
for img_id in os.listdir('cut_faces'):
    img = cv2.imread('cut_faces/' + img_id)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    
    cv2.imwrite('norm_cut_faces/' + img_id, img)
