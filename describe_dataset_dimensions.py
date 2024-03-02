from PIL import Image
import pandas as pd

from glob import glob
from tqdm import tqdm

pics = glob('./wiki_crop_all/*')

sizes = []
for pic in tqdm(pics):
    img = Image.open(pic)
    x = min(img.size)
    sizes.append(x)

print(pd.DataFrame(sizes).describe())
