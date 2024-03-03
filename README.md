# int20h_2024_shibarinu

```python.exe -m pip install --upgrade pip```

```pip install -r requirements.txt```

These are you go-to commands to install all the necessary packages for the project.

We don't really provide you with a script to run the project in one pipeline - but all the code <b>we</b> used to 
run everything can be found in this repo.

The dataset is manipulated and changed by files in dataset_manipulation folder;

Prepocessing scripts are in the preprocessing folders;

The main results are in .ipynb files (see clusterization, images with bounding boxes and such)


## Implementation

1. **Preprocessing**
- Classical traditional face detection - filtering out occlusions and small faces
- Mediapipe detection of face landmarks - face orientation
- Sorting images by best orientation, cleaning duplicates using landmarks, and using first 9000
- Cutting out face by landpoints contour
- Grayscale, histogram equalization - to balance skin color
- Square padding and resize to 100x100
2. **Encoding**
- DeepFace: VGG-Face, Facenet, DeepFace
- Fourier Transform with square and circle area of interest, different sizes
- Custom autoencoder [incomplete]
3. **Clustering**
- Dimensionality reduction - Principal Component Analysis
- KMeans, SpectralClustering with different n_clusters and metrics
- Age statistics, component assessment on the matter of age representability (ages parsed from file names, clusters measured for age std)
4. **Average**
- Taking closest to cluster center example
- Displaying fourier transform reconstruction of the cluster center
- Using a custom decoder [incomplete]
