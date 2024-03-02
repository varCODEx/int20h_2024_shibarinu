import os
import cv2
import mediapipe
import numpy as np
import pandas as pd

mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
masks = []

for img_path in os.listdir('reference_images'):
    img = cv2.imread('reference_images/' + img_path)
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        data = []
        for point in results.multi_face_landmarks[0].landmark:
            data.append([point.x, point.y, point.z])

        data = np.array(data)
        data = (data - data[0]) / np.max(np.abs(data), axis=0)

        masks.append(data)

masks = np.array(masks)

reference = masks.mean(axis=0)

print(reference.shape)

np.save('reference.npy', reference)