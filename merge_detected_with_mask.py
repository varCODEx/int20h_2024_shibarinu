import pandas as pd

# Open mask
mask_df = pd.read_csv('differences_with_ids.csv')

# Open detected faces
detected_faces_df = pd.read_csv('detected_faces.csv')

# Inner merge by image name
merged_df = pd.merge(mask_df, detected_faces_df, left_on='img_id', right_on='image_name', how='inner')
merged_df = merged_df.drop_duplicates(subset=['img_id'])
merged_df.drop(columns=['image_name'], inplace=True)
merged_df.reset_index(drop=True, inplace=True)

# Save
detected_faces_df.to_csv('merged_faces.csv', index=False)
