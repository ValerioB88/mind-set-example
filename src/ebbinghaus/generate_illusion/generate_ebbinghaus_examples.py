from src.ebbinghaus.generate_illusion.drawing_utils import DrawShape
from functools import partial
import numpy as np
import pathlib

ds = DrawShape(background='black', img_size=(224, 224), resize_up_resize_down=True)
n_small = 5
n_large = 8
ebb_large = partial(ds.create_ebbinghaus, r_c=0.0823, d=0.323, r2=0.15, n=n_small, shift=0)
ebb_small = partial(ds.create_ebbinghaus, r_c=0.0823, d=0.153, r2=0.05, n=n_large, shift=0)

num_rep = 15
folder = './data/ebbinghaus/'

subfolder = 'small'
shifts = np.linspace(0, np.rad2deg(np.pi*2/n_small), num_rep, endpoint=False, dtype=int)
pathlib.Path(folder + subfolder).mkdir(parents=True, exist_ok=True)
[ebb_small(shift=np.deg2rad(sh)).save(folder + subfolder + f'/s{idx}.png') for idx, sh in enumerate(shifts)]
print("Saved in " + folder + subfolder + f'/sX.png')

subfolder = 'large'
shifts = np.linspace(0, np.rad2deg(np.pi*2/n_large), num_rep, endpoint=False, dtype=int)
pathlib.Path(folder + subfolder).mkdir(parents=True, exist_ok=True)
[ebb_large(shift=np.deg2rad(sh)).save(folder + subfolder + f'/s{idx}.png') for idx, sh in enumerate(shifts)]
print("Saved in " + folder + subfolder + f'/sX.png')

##
ss = np.arange(-0.05, 0.055, 0.005)
[pathlib.Path(folder + f'{s:.3f}').mkdir(parents=True, exist_ok=True) for s in ss]

[[ds.create_ebbinghaus(r_c=0.0823+s).save(folder + f'{s:.3f}/s{idx}.png') for s in ss] for idx, sh in enumerate(shifts)]
print("Saved in " + folder + subfolder + f'/sX.png')
##

