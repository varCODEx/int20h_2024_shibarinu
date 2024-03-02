import cv2
import mediapipe
import numpy as np
import pandas as pd
import glob
import os
import tqdm

reference = np.load('reference.npy')
file_paths = glob.glob('wiki_crop_all/*')

mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
diffs = []

for img_id in tqdm.tqdm(file_paths):
    img = cv2.imread(img_id)
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        data = []
        for point in results.multi_face_landmarks[0].landmark:
            data.append([point.x, point.y, point.z])

        data = np.array(data)
        data = (data - data[0]) / np.max(np.abs(data), axis=0)

        diff = np.linalg.norm(reference - data)
        diffs.append([os.path.basename(img_id), diff])


differences = pd.DataFrame(diffs, columns=['img_id', 'difference'])
differences = differences.sort_values('difference', ascending=True)
differences.to_csv('img_enface_score.csv', index=False)