import re
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
import torch

def get_cossim_args(parser):
    parser.add_argument('--network_name', metavar='', help='[alexnet], [vgg11/16], [vgg11bn/16n/19bn], [resnet18/50/152], [inception_v3], [densenet121/201], [googlenet]', default='resnet152')
    parser.add_argument('--affine_transf_code', metavar='', help='specify what transformation to apply. Use t/s/r for translation, scale and rotation, and square brackets with the transformation limits. E.g. t[-0.2, 0.2]r[-50, 50]s will translate from -0.2 to 0.2 (normalize to image size), rotate from -50 to +50 degrees, and s indicates that will scale using default values. Default values are t=[-0.2, 0.2], r=0.7, 1.3], r=[0, 360].', default='')
    parser.add_argument('--affine_transf_background', metavar='', help='when rotating/scaling the image, a "fill in" background will be used. You can specify which one here: [black], [white], [random]<-for randomly pixellated, [cyan] etc.', default='black')
    parser.add_argument('--result_folder', metavar='', default='./results/')
    parser.add_argument('--repetitions', metavar='', help='How many times do we need to repeat a sample', default=50, type=int)
    parser.add_argument('--save_layers', metavar='', help='Specify against what layer we want to compute the cosine similarity analysis', default=['Conv2d', 'Linear'], nargs="+", type=str)
    parser.add_argument('--pretraining', metavar='', default='ImageNet', help='either [ImageNet] or [vanilla]')
    parser.add_argument('--matching_transform', metavar='', default=True, help='specify whether all members of a comparison set should have the same transformation applied to them (according to "affine_transf_code")')
    parser.add_argument('--use_cuda', default=torch.cuda.is_available(), type=lambda x: bool(int(x)), help='Whether to use GPU or not')
    return parser

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