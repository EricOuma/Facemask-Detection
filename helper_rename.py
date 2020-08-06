import pathlib
import os
dir = 'no_mask'

for count, name in enumerate(os.listdir(dir)):
    file_path = pathlib.Path(os.path.join(dir, name))
    file_ext = file_path.suffix
    os.rename(file_path, f'{dir}\\no_mask_{count}{file_ext}')
