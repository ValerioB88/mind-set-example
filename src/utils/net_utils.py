from typing import List

import numpy as np
from torchvision.transforms import transforms

import os
import torchvision
import torch.nn as nn
from sty import fg, ef, rs, bg
import torch.backends.cudnn as cudnn
import torch

from src.utils.callbacks import Callback, CallbackList


class GrabNet():

    @classmethod
    def get_net(cls, network_name, imagenet_pt=False, num_classes=None, **kwargs):
        """
        @num_classes = None indicates that the last layer WILL NOT be changed.
        """
        norm_values = dict(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        resize_value = 224

        if imagenet_pt:
            print(fg.red + "Loading ImageNet" + rs.fg)

        nc = 1000 if imagenet_pt else num_classes
        kwargs = dict(num_classes=nc) if nc is not None else dict()
        if network_name == 'vgg11':
            net = torchvision.models.vgg11(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg11bn':
            net = torchvision.models.vgg11_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16':
            net = torchvision.models.vgg16(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16bn':
            net = torchvision.models.vgg16_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg19bn':
            net = torchvision.models.vgg19_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'resnet18':
            net = torchvision.models.resnet18(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'resnet50':
            net = torchvision.models.resnet50(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'resnet152':
            net = torchvision.models.resnet152(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'alexnet':
            net = torchvision.models.alexnet(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'inception_v3':  # nope
            net = torchvision.models.inception_v3(pretrained=imagenet_pt, progress=True, **kwargs)
            resize_value = 299
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'densenet121':
            net = torchvision.models.densenet121(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'densenet201':
            net = torchvision.models.densenet201(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'googlenet':
            net = torchvision.models.googlenet(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            net = cls.get_other_nets(network_name, imagenet_pt, **kwargs)
            assert False if net is False else True, f"Network name {network_name} not recognized"

        return net, norm_values, resize_value

    @staticmethod
    def get_other_nets(network_name, num_classes, imagenet_pt, **kwargs):
        pass

class RandomPixels(torch.nn.Module):
    def __init__(self, background_color=(0, 0, 0), line_color=(255, 255, 255)):
        super().__init__()
        self.background_color = background_color
        self.line_color = line_color

    def forward(self, input):
        i = np.array(input)
        i = i.astype(np.int16)
        s_line = len(i[i == self.line_color])
        i[i == self.line_color] = np.repeat([1000, 1000, 1000], s_line/3, axis=0).flatten()

        s = len(i[i == self.background_color])
        i[i == self.background_color] = np.random.randint(0, 255, s)

        s_line = len(i[i == [1000, 1000, 1000]])
        i[i == [1000, 1000, 1000]] = np.repeat([0, 0, 0], s_line / 3, axis=0).flatten()
        i = i.astype(np.uint8)

        return transforms.ToPILImage()(i)


class RandomBackground(torch.nn.Module):
    def __init__(self, color_to_randomize=0):
        super().__init__()
        self.color_to_randomize = color_to_randomize

    def forward(self, input):
        i = np.array(input)
        s = len(i[i == self.color_to_randomize])

        i[i == self.color_to_randomize] = np.repeat([np.random.randint(0, 255, 3)], s/3, axis=0).flatten()
        return transforms.ToPILImage()(i)


def prepare_network(net, config, optimizer=None, train=True):
    pretraining_file = 'vanilla' if config.pretraining == 'ImageNet' else config.pretraining
    load_pretraining(net, pretraining_file, optimizer, config.use_cuda)
    net.cuda() if config.use_cuda else None

    cudnn.benchmark = True
    net.train() if train else net.eval()
    # print_net_info(net) if config.verbose else None


def load_pretraining(net, pretraining, optimizer=None, use_cuda=None):
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    if pretraining != 'vanilla':
        if os.path.isfile(pretraining):
            print(fg.red + f"Loading full model from {pretraining}..." + rs.fg, end="")
            ww = torch.load(pretraining, map_location=torch.device('cuda') if use_cuda else torch.device('cpu'))
            net.load_state_dict(ww['model'])
            print(fg.red + " Done." + rs.fg)

            if ww['optimizer'] and optimizer:
                print(fg.red + f"Loading optimizer state from {pretraining}..." + rs.fg, end="")
                optimizer.load_state_dict(ww['optimizer'])
                if use_cuda:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
                print(fg.red + " Done." + rs.fg)

        else:
            assert False, f"Pretraining path not found {pretraining}"


def print_net_info(net):
    num_trainable_params = 0
    tmp = ''
    print(fg.yellow)
    print("Params to learn:")
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            tmp += "\t" + name + "\n"
            print("\t" + name)
            num_trainable_params += len(param.flatten())
    print(f"Trainable Params: {num_trainable_params}")

    print('***Network***')
    print(net)
    print(ef.inverse + f"Network is in {('~train~' if net.training else '~eval~')} mode." + rs.inverse)
    print(rs.fg)
    print()


def make_cuda(fun, is_cuda):
    return fun.cuda() if is_cuda else fun


class Logs():
    value = None

    def __repl__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __copy__(self):
        return self.value

    def __deepcopy__(self, memodict={}):
        return self.value

    def __eq__(self, other):
        return self.value == other

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __radd__(self, other):
        return other + self.value

    def __rsub__(self, other):
        return other - self.value

    def __rfloordiv__(self, other):
        return other // self.value

    def __rtruediv__(self, other):
        return other / self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __floordiv__(self, other):
        return self.value // other

    def __truediv__(self, other):
        return self.value / other

    def __gt__(self, other):
        return self.value > other

    def __lt__(self, other):
        return self.value < other

    def __int__(self):
        return int(self.value)

    def __ge__(self, other):
        return self.value >= other

    def __le__(self, other):
        return self.value <= other

    def __float__(self):
        return float(self.value)

    def __pow__(self, power, modulo=None):
        return self.value ** power

    def __format__(self, format_spec):
        return format(self.value, format_spec)


class ExpMovingAverage(Logs):
    value = None

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def add(self, *args):
        if self.value is None:
            self.value = args[0]
        else:
            self.value = self.alpha * args[0] + (1 - self.alpha) * self.value
        return self


class CumulativeAverage(Logs):
    value = None
    n = 0

    def add(self, *args):
        if self.value is None:
            self.value = args[0]

        else:
            self.value = (args[0] + self.n * self.value) / (self.n+1)
        self.n += 1
        return self


def run(data_loader, use_cuda, net, callbacks: List[Callback] = None, optimizer=None, loss_fn=None, iteration_step=None, logs=None, **kwargs):
    if logs is None:
        logs = {}
    torch.cuda.empty_cache()

    make_cuda(net, use_cuda)

    callbacks = CallbackList(callbacks)
    callbacks.set_model(net)
    callbacks.set_loss_fn(loss_fn)

    callbacks.on_train_begin()

    tot_iter = 0
    epoch = 0
    logs['tot_iter'] = 0
    while True:
        callbacks.on_epoch_begin(epoch)
        logs['epoch'] = epoch
        for batch_index, data in enumerate(data_loader, 0):
            callbacks.on_batch_begin(batch_index, logs)
            loss, y_true, y_pred, logs = iteration_step(data, net, loss_fn, optimizer, use_cuda, logs, **kwargs)
            logs.update({
                # 'y_pred': y_pred,
                'loss': loss.item(),
                # 'y_true': y_true,
                'tot_iter': tot_iter,
                'stop': False})

            callbacks.on_training_step_end(batch_index, logs)
            callbacks.on_batch_end(batch_index, logs)
            if logs['stop']:
                break
            tot_iter += 1

        callbacks.on_epoch_end(epoch, logs)
        epoch += 1
        if logs['stop']:
            break

    callbacks.on_train_end(logs)
    return net, logs