import cv2
import mediapipe
import numpy as np
import pandas as pd
import glob
import os
import tqdm
import matplotlib.pyplot as plt

differences = pd.read_csv('img_enface_score.csv').iloc[:9]

mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

for index, row in tqdm.tqdm(differences.iterrows(), total=differences.shape[0]):
    img = cv2.imread(row['img_id'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img)
    landmarks = results.multi_face_landmarks[0]

    df = pd.DataFrame(list(mp_face_mesh.FACEMESH_FACE_OVAL), columns = ["p1", "p2"])

    routes_idx = []

    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]

    for i in range(0, df.shape[0]):
        
        #print(p1, p2)
        
        obj = df[df["p1"] == p2]
        p1 = obj["p1"].values[0]
        p2 = obj["p2"].values[0]
        
        route_idx = []
        route_idx.append(p1)
        route_idx.append(p2)
        routes_idx.append(route_idx)

    routes = []

    #for source_idx, target_idx in mp_face_mesh.FACEMESH_FACE_OVAL:
    for source_idx, target_idx in routes_idx:
        
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]
            
        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

        #cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)
        
        routes.append(relative_source)
        routes.append(relative_target)


    mask = np.zeros((img.shape[0], img.shape[1]))
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)
    
    out = np.zeros_like(img)
    out[mask] = img[mask]

    x1 = min(routes, key=lambda x: x[1])[1]
    x2 = max(routes, key=lambda x: x[1])[1]
    y1 = min(routes, key=lambda x: x[0])[0]
    y2 = max(routes, key=lambda x: x[0])[0]

    cropped_img = out[x1:x2, y1:y2]

    cv2.imwrite('cut_faces/' + os.path.basename(row['img_id']), cropped_img[:, :, ::-1])