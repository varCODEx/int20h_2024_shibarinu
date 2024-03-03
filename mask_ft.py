import os
from tqdm import tqdm

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def create_circular_mask(shape, center, radius):
    """
    Create a circular mask.

    Parameters:
    - shape: tuple, shape of the mask (rows, cols)
    - center: tuple, center of the circle (x, y)
    - radius: int, radius of the circle

    Returns:
    - mask: 2D boolean array representing the circular mask
    """
    rows, cols = shape
    x, y = center

    # Create a grid of coordinates
    Y, X = np.ogrid[:rows, :cols]

    # Calculate the distance from each pixel to the center
    distance_from_center = np.sqrt((X - x)**2 + (Y - y)**2)

    # Create the circular mask
    mask = distance_from_center <= radius

    return mask


files = os.listdir('/kaggle/input/cut-faces-top-9000/norm_cut_faces')
masked_ft_df = []

for image_filename in tqdm(files, unit='image', desc='Processing images...'):
    image = plt.imread("/kaggle/input/cut-faces-top-9000/norm_cut_faces/" + image_filename)


    # Calculate the 2D Fourier transform
    ft = calculate_2dft(image)
    # Example shape of the mask (should match the shape of the Fourier transform)
    mask_shape = ft.shape

    # Example center and radius of the circle
    center = (mask_shape[0] // 2, mask_shape[1] // 2)
    radius = 8

    # Create the circular mask
    mask = create_circular_mask(mask_shape, center, radius)

    # Apply the mask to the Fourier transform
    ft_masked = ft * mask

    # Find the indices where ft_masked is non-zero
    nonzero_indices = np.nonzero(ft_masked)

    # Get the bounding box of the non-zero region
    min_row = np.min(nonzero_indices[0])
    max_row = np.max(nonzero_indices[0])
    min_col = np.min(nonzero_indices[1])
    max_col = np.max(nonzero_indices[1])

    # Extract the non-zero region from ft_masked
    ft_masked_nonzero = ft_masked[min_row:max_row+1, min_col:max_col+1]

    masked_ft_df.append({'image_name': image_filename, 'ft': ft_masked_nonzero})


masked_ft_df = pd.DataFrame(masked_ft_df)
masked_ft_df.to_csv('masked_ft.csv', index=False)
