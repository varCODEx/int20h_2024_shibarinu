import shutil

from pathlib import Path
from glob import glob

from tqdm import tqdm

df = Path('./wiki_crop')
dest = Path('./wiki_crop_all')
dest.mkdir(exist_ok=True)

folders = [folder for folder in glob(str(df / '*')) if Path(folder).is_dir()]

files = []
for folder in tqdm(folders, unit='folder', desc='Copying folders content to dest folder...'):
    folder_files = glob(str(Path(folder) / '*'))
    files.extend(folder_files)

    shutil.copytree(folder, dest, dirs_exist_ok=True)

assert len(files) == len(files)
## 62328
