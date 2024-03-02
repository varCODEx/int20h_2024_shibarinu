import cv2 as cv
from tqdm import tqdm
import os
import pandas as pd

# Load images folder
files = os.listdir('/kaggle/working/wiki_crop_all')
image_files = [file for file in files]
detected_faces = []

# Load the classifier and create a cascade object for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Iterate through images
for image in tqdm(image_files, unit='image', desc='Processing images...'):
    # Open original image
    original_image = cv.imread('/kaggle/working/wiki_crop_all/' + image)

    if original_image is None:
            print(f"Error: Failed to read image '{image}'. Skipping...")
            continue

    # Gray scale image
    grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    # Get cords of face
    detected_face = face_cascade.detectMultiScale(grayscale_image)
    # Save cords
    for (x, y, w, h) in detected_face:
        detected_faces.append({'image_name': image, 'cords': (x, y, w, h)})

# Save result
detected_faces_df = pd.DataFrame(detected_faces)
detected_faces_df.to_csv('/kaggle/workin/detected_faces.csv', index=False)
