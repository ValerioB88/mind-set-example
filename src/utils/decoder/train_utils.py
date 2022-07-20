import torch
import torchvision
from torch import nn as nn
from src.utils.net_utils import make_cuda
from src.utils.misc import imshow_batch

def decoder_step(data, model, loss_fn, optimizers, use_cuda, logs, logs_prefix, train, method, **kwargs):
    num_decoders = len(model.decoders)

    images, labels = data
    images = make_cuda(images, use_cuda)
    labels = make_cuda(labels, use_cuda)
    if train:
        [optimizers[i].zero_grad() for i in range(num_decoders)]
    out_dec = model(images)
    loss = make_cuda(torch.tensor([0.], requires_grad=True), use_cuda)
    loss_decoder = []
    for idx, od in enumerate(out_dec):
        loss_decoder.append(loss_fn(od,
                labels))
        loss += loss_decoder[-1]

    logs[f'{logs_prefix}ema_loss'].add(loss.item())

    if method == 'regression':
        if train:
            [logs[f'{logs_prefix}ema_rmse_{idx}'].add(torch.sqrt(ms).item()) for idx, ms in enumerate(loss_decoder)]
        else:
            [logs[f'{logs_prefix}rmse_{idx}'].add(torch.sqrt(ms).item()) for idx, ms in enumerate(loss_decoder)]

            logs[f'{logs_prefix}rmse'].add(torch.sqrt(loss / num_decoders).item())

    elif method == 'classification':
        if train:
            [logs[f'{logs_prefix}ema_acc_{idx}'].add(torch.mean((torch.argmax(out_dec[idx], 1) == labels).float()).item()) for idx in range(len(loss_decoder))]
        else:
            [logs[f'{logs_prefix}acc_{idx}'].add(torch.mean((torch.argmax(out_dec[idx], 1) == labels).float()).item()) for idx in range(len(loss_decoder))]
            logs[f'{logs_prefix}acc'].add(torch.mean(torch.tensor([torch.mean((torch.argmax(out_dec[idx], 1) == labels).float()).item() for idx in range(len(loss_decoder))])).item())


    if 'collect_data' in kwargs and kwargs['collect_data']:
        logs['data'] = data
    if train:
        loss.backward()
        [optimizers[i].step() for i in range(num_decoders)]



def config_to_path_train(config):
    return f"/ebbinghaus/decoder/{config.network_name}"


class ResNet152decoders(nn.Module):
    def __init__(self, imagenet_pt, num_outputs=1, **kwargs):
        super().__init__()
        self.net = torchvision.models.resnet152(pretrained=imagenet_pt, progress=True, **kwargs)
        self.decoders = nn.ModuleList([nn.Linear(224*224*3, num_outputs),
                                      nn.Linear(802816, num_outputs),
                                      nn.Linear(401408, num_outputs),
                                      nn.Linear(200704, num_outputs),
                                      nn.Linear(100352, num_outputs),
                                      nn.Linear(2048, num_outputs)])

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