import torch
import torchvision
from torch import nn as nn
from src.utils.net_utils import make_cuda
from src.utils.misc import imshow_batch

def decoder_step(data, model, loss_fn, optimizers, use_cuda, logs, train, decoder_to_train=-1, **kwargs):
    """
    :param decoder_to_train: if decoder to train is -1, accumulate the loss across all decoders
    :return:
    """
    num_decoders = len(model.decoders)

    images, labels = data
    images = make_cuda(images, use_cuda)
    labels = make_cuda(labels, use_cuda)
    if train:
        [optimizers[i].zero_grad() for i in range(num_decoders)]
    out_dec = model(images)
    loss = make_cuda(torch.tensor([0.], requires_grad=True), use_cuda)
    mse_dec = []
    for idx, od in enumerate(out_dec):
        mse_dec.append(loss_fn(od,
                labels))
        loss += mse_dec[-1]
    predicted = torch.cat(out_dec)
    ground = labels.repeat(len(out_dec))

    logs['ema_loss'].add(loss.item())

    [logs[f'ema_rmse_{idx}'].add(torch.sqrt(ms)) for idx, ms in enumerate(mse_dec)]
    logs[f'ca_rmse'].add(torch.sqrt(loss/num_decoders).item())

    logs['y_true'] = ground
    logs['y_pred'] = predicted
    if 'collect_data' in kwargs and kwargs['collect_data']:
        logs['data'] = data
    if train:
        loss.backward()
        [optimizers[i].step() for i in range(num_decoders)]

    return loss, ground, predicted, logs


def config_to_path_train(config):
    return f"/ebbinghaus/decoder/{config.network_name}"


class ResNet152decoders(nn.Module):
    def __init__(self, imagenet_pt, **kwargs):
        super().__init__()
        self.net = torchvision.models.resnet152(pretrained=imagenet_pt, progress=True, **kwargs)
        self.decoders = nn.ModuleList([nn.Linear(224*224*3, 1),
                                      nn.Linear(802816, 1),
                                      nn.Linear(401408, 1),
                                      nn.Linear(200704, 1),
                                      nn.Linear(100352, 1),
                                      nn.Linear(2048, 1)])

    def forward(self, x):
        out_dec = []
        out_dec.append(self.decoders[0](torch.flatten(x, 1)).squeeze())

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        out_dec.append(self.decoders[1](torch.flatten(x, 1)).squeeze())

        x = self.net.layer2(x)
        out_dec.append(self.decoders[2](torch.flatten(x, 1)).squeeze())

        x = self.net.layer3(x)
        out_dec.append(self.decoders[3](torch.flatten(x, 1)).squeeze())

        x = self.net.layer4(x)
        out_dec.append(self.decoders[4](torch.flatten(x, 1)).squeeze())

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        out_dec.append(self.decoders[5](torch.flatten(x, 1)).squeeze())
        return out_dec