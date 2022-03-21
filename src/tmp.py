import re
import glob
import os
import shutil
import PIL.Image as Image
all_files = glob.glob('./data/NAPvsMP/**')
i = all_files[0]
for i in all_files:
    name, type = [int(i) for i in re.search('([\d\w]+)_([\d\w]+).', os.path.basename(i)).groups()]
    if type == 0:
        Image.open(i).convert('RGB').save(os.path.dirname(i) + f'/new/{name}_base.png')
    if type == 1:
        Image.open(i).convert('RGB').save(os.path.dirname(i) + f'/new/{name}_NAP.png')
    if type == 2:
        Image.open(i).convert('RGB').save(os.path.dirname(i) + f'/new/{name}_MP.png')
# all_files = glob.glob('./data/NAPvsMPsilh/VogelsSil124_files/**')
# i = all_files[0]
# for i in all_files:
#     obj_name = [int(i) for i in re.search(r'Obj(\d+)', i).groups()[0]][0]
#     if 'Basic' in i:
#         Image.open(i).convert('RGB').save(os.path.dirname(os.path.dirname(i)) + f'/{obj_name}_0.png')
#     elif 'NAP' in i:
#         Image.open(i).convert('RGB').save(os.path.dirname(os.path.dirname(i)) + f'/{obj_name}_1.png')
#     elif 'Metric' in i:
#         Image.open(i).convert('RGB').save(os.path.dirname(os.path.dirname(i)) + f'/{obj_name}_2.png')
#



for i in all_files:
    set, alt = [int(i) for i in re.search('(\d+)ol(\d+).bmp', os.path.basename(i)).groups()]
    if alt == 2:
        alt = 0
    elif alt == 0:
        alt = 1
    elif alt == 4:
        alt = 2
    import shutil

    Image.open(i).convert('RGB').save(os.path.dirname(os.path.dirname(i)) + f'/{set}_{alt}.png')
