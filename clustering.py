import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Extract magnitudes and flatten the arrays
data = []
for ft_array in tqdm(masked_ft_df['ft'], unit='tf', desc='Processing tf...'):
    magnitudes = np.abs(ft_array)
    flattened_array = magnitudes.flatten()
    data.append(flattened_array)

# Convert the list of arrays into a 2D feature matrix
feature_matrix = np.array(data)

kmeans = KMeans(n_clusters=7)  # Choose the number of clusters as needed
kmeans.fit(feature_matrix)

# Assign cluster labels to each image
cluster_labels = kmeans.labels_

# Add cluster labels to the DataFrame
masked_ft_df['cluster_label'] = cluster_labels

# Convert the 'ft' column to a numpy array
masked_ft_df['ft'] = masked_ft_df['ft'].apply(np.array)

# Group the DataFrame by cluster_label and calculate the mean of the 'ft' column for each group
average_ft_per_cluster = masked_ft_df.groupby('cluster_label')['ft'].apply(np.mean)


def calculate_ncc(ft1, ft2):
    # Calculate the NCC between two Fourier transforms
    ncc = np.corrcoef(np.abs(ft1.flatten()), np.abs(ft2.flatten()))[0, 1]
    return ncc

# Iterate through cluster items and find the top 3 images closest to the average
top_images = []

for cluster_label, average_ft in average_ft_per_cluster.items():
    cluster_df = masked_ft_df[masked_ft_df['cluster_label'] == cluster_label]
    distances = []

    # Calculate NCC for each image in the cluster
    for ft in cluster_df['ft']:
        ncc = calculate_ncc(np.abs(ft), np.abs(average_ft))
        distances.append(ncc)

    # Get the indices of the top 3 images with the highest NCC values
    top_indices = np.argsort(distances)[-5:]

    # Get the image names corresponding to the top indices
    top_image_names = cluster_df.iloc[top_indices]['image_name'].tolist()

    # Append the top image names to the list
    top_images.extend(top_image_names)

# Define the number of clusters
num_clusters = len(top_images) // 5

# Iterate through clusters and display top 3 images for each cluster
for i in range(num_clusters):
    cluster_images = top_images[i * 5: (i + 1) * 5]
    plt.figure(figsize=(15, 5))
    #plt.set_cmap("gray")

    # Iterate through images in the cluster
    for j, image_name in enumerate(cluster_images, start=1):
        image_filename = "/kaggle/working/wiki_crop_all/" + image_name
        image = plt.imread(image_filename)

        # Plot the image in the corresponding subplot
        plt.subplot(1, 5, j)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Image {j}")

    plt.show()
