import pandas as pd

# Open mask
mask_df = pd.read_csv('../data/img_enface_score.csv')

# Open detected faces
detected_faces_df = pd.read_csv('../data/detected_faces.csv')

detected_faces_df['str_coords'] = detected_faces_df['cords'].apply(lambda x: str(x))
detected_faces_df.drop_duplicates(subset=['image_name', 'str_coords'], inplace=True)
detected_faces_df.drop_duplicates(subset=['image_name'], inplace=True, keep=False)

# Inner merge by image name
merged_df = pd.merge(mask_df, detected_faces_df, left_on='img_id', right_on='image_name', how='inner')
merged_df = merged_df.drop_duplicates(subset=['img_id'])

print(len(detected_faces_df), len(mask_df), len(merged_df))

merged_df = merged_df.drop_duplicates(subset=['difference', 'str_coords'])
print(len(merged_df))

merged_df = merged_df.drop(columns=['str_coords', 'image_name'])
merged_df.reset_index(drop=True, inplace=True)

print(merged_df.columns)
print(merged_df.head())

# Save
merged_df.to_csv('../data/merged_faces.csv', index=False)
