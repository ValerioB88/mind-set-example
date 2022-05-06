import torch
import sty
import numpy as np
from typing import List
from src.utils.net_utils import make_cuda
from copy import deepcopy


class RecordActivations:
    def __init__(self, net, use_cuda=None, only_save: List[str] = None, detach_tensors=True):
        if only_save is None:
            self.only_save = ['Conv2d', 'Linear']
        else:
            self.only_save = only_save
        self.cuda = False
        if use_cuda is None:
            if torch.cuda.is_available():
                self.cuda = True
            else:
                self.cuda = False
        else:
            self.cuda = use_cuda
        self.net = net
        self.detach_tensors = detach_tensors
        self.activation = {}
        self.last_linear_layer = ''
        self.all_layers_names = []
        self.setup_network()


    def setup_network(self):
        self.was_train = self.net.training
        self.net.eval()  # a bit dangerous
        print(sty.fg.yellow + "Network put in eval mode in Record Activation" + sty.rs.fg)
        all_layers = self.group_all_layers()
        self.hook_lists = []
        for idx, i in enumerate(all_layers):
            name = '{}: {}'.format(idx, str.split(str(i), '(')[0])
            if np.any([ii in name for ii in self.only_save]):
                ## Watch out: not all of these layers will be used. Some networks have conditional layers depending on training/eval mode. The best way to get the right layers is to check those that are returned in "activation"
                self.all_layers_names.append(name)
                self.hook_lists.append(i.register_forward_hook(self.get_activation(name)))
        self.last_linear_layer = self.all_layers_names[-1]

    def get_activation(self, name):
        def hook(model, input, output):
                self.activation[name] = (output.detach() if self.detach_tensors else output)
        return hook

    def group_all_layers(self):
        all_layers = []

        def recursive_group(net):
            for layer in net.children():
                if not list(layer.children()):  # if leaf node, add it to list
                    all_layers.append(layer)
                else:
                    recursive_group(layer)

        recursive_group(self.net)
        return all_layers

    def remove_hooks(self):
        for h in self.hook_lists:
            h.remove()
        if self.was_train:
            self.net.train()


class RecordCossim(RecordActivations):
    def compute_cosine_pair(self, image0, image1):# path_save_fig, stats):
        cossim = {}

        self.net(make_cuda(image0.unsqueeze(0), torch.cuda.is_available()))
        first_image_act = {}
        activation_image1 = deepcopy(self.activation)
        for name, features1 in self.activation.items():
            if not np.any([i in name for i in self.only_save]):
                continue
            first_image_act[name] = features1.flatten()

        self.net(make_cuda(image1.unsqueeze(0), torch.cuda.is_available()))
        activation_image2 = deepcopy(self.activation)

        second_image_act = {}
        for name, features2 in self.activation.items():
            if not np.any([i in name for i in self.only_save]):
                continue
            second_image_act[name] = features2.flatten()
            if name not in cossim:
                cossim[name] = []
            cossim[name].append(torch.nn.CosineSimilarity(dim=0)(first_image_act[name], second_image_act[name]).item())

        return cossim

