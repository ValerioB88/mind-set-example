import numpy as np
import torch
import matplotlib.pyplot as plt
import re
import torchvision.transforms.functional as F

class Config:
    def __init__(self, **kwargs):
        self.use_cuda = torch.cuda.is_available()
        self.verbose = True
        [self.__setattr__(k, v) for k, v in kwargs.items()]

    def __setattr__(self, *args, **kwargs):
        super().__setattr__(*args, **kwargs)

def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image



def save_figs(path, set, extra_info='', n=None):
    fig, ax = plt.subplots(len(set) if n is None else n, 2)
    if np.ndim(ax) == 1:
        ax = np.array([ax])
    for idx, axx in enumerate(ax):
        axx[0].imshow(set[idx][0])
        axx[1].imshow(set[idx][1])
    # [x.axis('off') for x in ax.flatten()]
    plt.gcf().set_size_inches([2.4, 5])
    plt.suptitle(f'Size: f{set[0][0].shape}\n{extra_info}')

    [x.set_xticks([]) for x in ax.flatten()]
    [x.set_yticks([]) for x in ax.flatten()]

    plt.savefig(path)


def get_new_affine_values(transf_code):
    # Example transf_code = 's[0.2, 0.3]tr[0,90]'  -> s is within 0.2, 0.3, t is default, r is between 0 and 90 degrees
    def get_values(code):
        real_num = r'[-+]?[0-9]*\.?[0-9]+'
        try:
            return [float(i) for i in re.search(f'{code}\[({real_num}),\s?({real_num})]', transf_code).groups()]
        except AttributeError:
            if code == 't':
                return [-0.2, 0.2]
            if code == 's':
                return [0.7, 1.3]
            if code == 'r':
                return [0, 360]

    tr = [np.random.uniform(*get_values('t')), np.random.uniform(*get_values('t'))] if 't' in transf_code else (0, 0)
    scale = np.random.uniform(*get_values('s')) if 's' in transf_code else 1.0
    rot = np.random.uniform(*get_values('r')) if 'r' in transf_code else 0
    return {'rt': rot, 'tr': tr, 'sc': scale, 'sh': 0.0}


def my_affine(img, translate, **kwargs):
    return F.affine(img, translate=[int(translate[0] * img.size[0]), int(translate[1] * img.size[1])], **kwargs)
