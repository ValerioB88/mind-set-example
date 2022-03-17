import torch
import sty
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from tqdm import tqdm
from typing import List


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


