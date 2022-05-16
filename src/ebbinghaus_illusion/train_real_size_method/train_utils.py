import torch
import torchvision
from torch import nn as nn
from src.utils.net_utils import make_cuda


def regression_step(data, model, loss_fn, optimizer, use_cuda, logs, train, **kwargs):
    images, labels = data
    images = make_cuda(images, use_cuda)
    labels = make_cuda(labels.to(torch.float32), use_cuda)
    optimizer.zero_grad() if train else None

    predicted = model(images)
    loss = loss_fn(predicted.squeeze(), labels)
    logs['ema_loss'].add(loss.item())
    logs[f'ema_rmse'].add(torch.sqrt(loss))
    logs[f'ema_mse'].add(loss)
    logs[f'ca_mse'].add(loss)

    logs['y_true'] = labels
    logs['y_pred'] = predicted.squeeze()
    if 'collect_data' in kwargs and kwargs['collect_data']:
        logs['data'] = data
    if train:
        loss.backward()
        optimizer.step()

    return loss, labels, predicted, logs


class ResNet152_size(nn.Module):
    def __init__(self, imagenet_pt, **kwargs):
        super().__init__()
        self.net = torchvision.models.resnet152(pretrained=imagenet_pt, progress=True, **kwargs)
        self.output = nn.Linear(2048, 1)

    def forward(self, x):
        out_dec = []
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x