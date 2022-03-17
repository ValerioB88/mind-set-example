
import re
import glob
import os
all_files = glob.glob('./data/NAPvsMP/files/**')
for i in all_files:
    set, alt = [int(i) for i in re.search('(\d+)os(\d+).bmp', os.path.basename(i)).groups()]
    if alt == 2:
        alt = 0
    elif alt == 0:
        alt = 1
    elif alt == 4:
        alt = 2
    import shutil

    shutil.copy(i, os.path.dirname(os.path.dirname(i)) + f'/{set}_{alt}.bmp')
