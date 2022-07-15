"""
General training script for decoder approach. The only thing you need to change is the loading `DATASET` variable. Note that in this case the EbbinghausTrain dataset is always generated on the fly (but you could specify the kwarg "path" to save/load it on disk).
"""
# from src.utils.Config import Config
from src.utils.decoder.train_utils import decoder_step, ResNet152decoders
from src.utils.net_utils import prepare_network, ExpMovingAverage, CumulativeAverage, run
from glob import glob
from torch.utils.data import DataLoader
from src.utils.misc import weblog_dataset_info
from src.utils.callbacks import *
import torchvision
from src.utils.decoder.data_utils import RegressionDataset
import argparse
import torch.backends.cudnn as cudnn
from src.utils.net_utils import load_pretraining
from functools import partial

def fix_dataset(dataset, name_ds=''):
    dataset.name_ds = name_ds
    dataset.stats = {'mean': [0.491, 0.482, 0.44], 'std': [0.247, 0.243, 0.262]}
    add_resize = False
    if next(iter(dataset))[0].size[0] != 244:
        add_resize = True

    dataset.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=dataset.stats['mean'],
                                                         std=dataset.stats['std'])])
    if add_resize:
        dataset.transform.transforms.insert(0, torchvision.transforms.Resize(224))
    return dataset


parser =argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test_results_folder', metavar='', default='./results/tmp/')
parser.add_argument('--model_output_path', metavar='')
parser.add_argument('--train_dataset', metavar='', help='Either a folder of folders (one for each class) for classification - or a single folder with all the images for regression (see README.md for more info)')
parser.add_argument('--test_datasets', nargs='+', help='same format as train_dataset. You can spacify multiple folders. Default is [] (empty).', default=[])


parser.add_argument('--gpu_num', metavar='', help='what gpu to use (int, starts at 0).', type=int, default=0)
parser.add_argument('--batch_size', metavar='', type=int, default=64)
parser.add_argument('--continue_train', metavar='', help='continue training using model specified here. Note that the optimizer state is NOT saved, which might affect performance for the first epoch or this training.', default=False)
parser.add_argument('--stop_at_epoch', metavar='', default=50, help='stop after the specified number of epochs', type=int)
parser.add_argument('--stop_at_loss', metavar='', default=None, help='stop when loss is lower than specified value (for some time)')
parser.add_argument('--learning_rate', metavar='', default=1e-5, type=float)
parser.add_argument('--weight_decay', metavar='', default=0.0, type=float)
parser.add_argument('--stop_at_accuracy', metavar='', default=None, help='stop when accuracy reaches and stays for a bit on the specified value. Doesn\'t make sense with regression. With regression, use \'stop_at_loss\' instead', type=float)
parser.add_argument('--neptune_proj_name', metavar='', default=False,  help='if you want to log in neptune.ai, specify the project name. Really useful, but you need an account there + you need to have created a project with the correct name through their UI + You need to set up your API token: \nhttps://docs.neptune.ai/getting-started/installation\nIf you DO use Neptune I automatically log many things including debug images through this script.')

config = parser.parse_known_args()[0]
# config = parser.parse_known_args(['--model_output_path', './models/ebbinghaus/prova.pt', '--train_dataset', './data/ebbinghaus/train_random_data', '--test_datasets',
#     './data/ebbinghaus/test_random_data',
#     './data/ebbinghaus/test_small_flankers_data',
#     './data/ebbinghaus/test_big_flankers_data', '--neptune_proj_name', 'Ebbinghaus'])[0]
#
# config = parser.parse_known_args(['--model_output_path', './models/ebbinghaus/prova.pt', '--train_dataset', './data/miniMNIST/training/', '--test_data', './data/miniMNIST/testing1/'])[0]
#
config.train_dataset = config.train_dataset.rstrip('/')
config.test_datasets = [i.rstrip('/') for i in config.test_datasets]

config.weblogger = False
if config.neptune_proj_name:
    try:
        neptune_run = neptune.init(f'valeriobiscione/{config.neptune_proj_name}')
        # neptune_run["sys/tags"].add(list_tags)
        neptune_run["parameters"] = config.dict.keys()
        config.weblogger = neptune_run
        def __setattr__(self, *args, **kwargs):
            if isinstance(self.weblogger, neptune.Run):
                self.weblogger[f"parameters/{args[0]}"] = str(args[1])
            super().__setattr__(*args, **kwargs)
        # In this way, everything added from now on to args will be automatically logged.
        config.__setattr__ = __setattr__

    except:
        print("Initializing neptune didn't work, maybe you don't have the neptune client installed or you haven't set up the API token (https://docs.neptune.ai/getting-started/installation). Neptune logging won't be used :(")

config.use_cuda = torch.cuda.is_available()
config.is_pycharm = True if 'PYCHARM_HOSTED' in os.environ else False
torch.cuda.set_device(config.gpu_num) if torch.cuda.is_available() else None

if not list(os.walk(config.train_dataset))[0][1]:
    print(sty.fg.yellow + sty.ef.inverse + 'You pointed to a folder containin only images, which means that you are going to run a REGRESSION method' + sty.rs.ef)
    config.method = 'regression'
else:
    print(sty.fg.yellow + sty.ef.inverse + 'You pointed to a dataset of folders, which means that you are going to run a CLASSIFICATION method' + sty.rs.ef)
    config.method = 'classification'


[print(fg.red + f'{i[0]}:' + fg.blue + f' {i[1]}' + rs.fg) for i in config._get_kwargs()]

from torchvision.datasets import ImageFolder
ds = ImageFolder if config.method == 'classification' else RegressionDataset
train_dataset = fix_dataset(ds(root=config.train_dataset), name_ds=os.path.basename(config.train_dataset))


config.net = ResNet152decoders(imagenet_pt=True, num_outputs=1 if config.method == 'regression' else len(train_dataset.classes))
num_decoders = len(config.net.decoders)

if config.continue_train:
    config.pretraining = config.continue_train
    load_pretraining(config.net, config.pretraining, optimizer=None, use_cuda=config.use_cuda)

print(sty.ef.inverse + "FREEZING CORE NETWORK" + sty.rs.ef)
for param in config.net.parameters():
    param.requires_grad = False
for param in config.net.decoders.parameters():
    param.requires_grad = True

config.net.cuda() if config.use_cuda else None
cudnn.benchmark = True
config.net.train()

config.loss_fn = torch.nn.MSELoss() if config.method == 'regression' else torch.nn.CrossEntropyLoss()
config.optimizers = [torch.optim.Adam(config.net.decoders[i].parameters(),
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay) for i in range(num_decoders)]


train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          drop_last=True,
                          shuffle=True,
                          num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                          timeout=0 if config.use_cuda and not config.is_pycharm else 0,
                          pin_memory=True)

weblog_dataset_info(train_loader, weblogger=config.weblogger, num_batches_to_log=1, log_text='train') if config.weblogger else None

ds_type = RegressionDataset if config.method == 'regression' else ImageFolder
test_datasets = [fix_dataset(ds_type(root=path), name_ds=os.path.splitext(os.path.basename(path))[0]) for path in config.test_datasets]

test_loaders = [DataLoader(td,
                           batch_size=config.batch_size,
                           drop_last=True,
                           num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                           timeout=0 if config.use_cuda and not config.is_pycharm else 0,
                           pin_memory=True) for td in test_datasets]

[weblog_dataset_info(td, weblogger=config.weblogger, num_batches_to_log=1, log_text='test') for td in test_loaders]

config.step = decoder_step


def call_run(loader, train, callbacks, method, logs_prefix='', logs=None, **kwargs):
    if logs is None:
        logs = {}
    logs.update({f'{logs_prefix}ema_loss': ExpMovingAverage(0.2)})

    if train:
        logs.update({f'{logs_prefix}ema_{log_type}_{i}': ExpMovingAverage(0.2) for i in range(6)})
    else:
        logs.update({f'{logs_prefix}{log_type}': CumulativeAverage()})
        logs.update({f'{logs_prefix}{log_type}_{i}': CumulativeAverage() for i in range(6)})

    return run(loader,
               use_cuda=config.use_cuda,
               net=config.net,
               callbacks=callbacks,
               loss_fn=config.loss_fn,
               optimizer=config.optimizers,
               iteration_step=config.step,
               train=train,
               logs=logs,
               logs_prefix=logs_prefix,
               collect_data=kwargs.pop('collect_data', False),
               stats=train_dataset.stats,
               method=method)

def stop(logs, cb):
    logs['stop'] = True
    print('Early Stopping')

log_type = 'acc' if config.method == 'classification' else 'rmse' # rmse: Root Mean Square Error : sqrt(MSE)
all_cb = [
    StopFromUserInput(),
    ProgressBar(l=len(train_dataset), batch_size=config.batch_size, logs_keys=['ema_loss',
                                                                               *[f'ema_{log_type}_{i}' for i in range(num_decoders)]]),
    PrintNeptune(id='ema_loss', plot_every=10, weblogger=config.weblogger),
    *[PrintNeptune(id=f'ema_{log_type}_{i}', plot_every=10, weblogger=config.weblogger) for i in range(num_decoders)],
    # Either train for X epochs
    TriggerActionWhenReachingValue(mode='max', patience=1, value_to_reach=config.stop_at_epoch, check_after_batch=False, metric_name='epoch', action=stop, action_name=f'{config.stop_at_epoch} epochs'),


    *[DuringTrainingTest(testing_loaders=tl, eval_mode=False, every_x_epochs=1, auto_increase=False, weblogger=config.weblogger, log_text='test during train TRAINmode', use_cuda=config.use_cuda, logs_prefix=f'{tl.dataset.name_ds}/', call_run=partial(call_run, method=config.method), plot_samples_corr_incorr=False, callbacks=[
        SaveInfoCsv(log_names=['epoch', *[f'{tl.dataset.name_ds}/{log_type}_{i}' for i in range(num_decoders)]], path=config.test_results_folder + f'/{tl.dataset.name_ds}.csv'),
        # if you don't use neptune, this will be ignored
        PrintNeptune(id=f'{tl.dataset.name_ds}/{log_type}', plot_every=np.inf, log_prefix='test_TRAIN', weblogger=config.weblogger),
        PrintConsole(id=f'{tl.dataset.name_ds}/{log_type}', endln=" -- ", plot_every=np.inf, plot_at_end=True),
        *[PrintConsole(id=f'{tl.dataset.name_ds}/{log_type}_{i}', endln=" "
                                                       "/ ", plot_every=np.inf, plot_at_end=True) for i in range(num_decoders)],
                         ]) for tl in test_loaders]
]

all_cb.append(SaveModel(config.net, config.model_output_path, loss_metric_name='ema_loss')) if config.model_output_path and not config.is_pycharm else None
all_cb.append(
    TriggerActionWhenReachingValue(mode='min', patience=20, value_to_reach=config.stop_at_loss, check_every=10, metric_name='ema_loss', action=stop, action_name=f'goal [{config.stop_at_loss}]')) if config.stop_at_loss else None

net, logs = call_run(train_loader, True, all_cb, config.method)
config.weblogger.stop() if config.weblogger else None
