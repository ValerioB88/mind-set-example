import torch.nn

from src.ebbinghaus_illusion.train_real_size_method.train_utils import regression_step, ResNet152_size
from src.utils.Config import Config
from src.utils.net_utils import prepare_network, ExpMovingAverage, CumulativeAverage, run
from torch.utils.data import DataLoader
from src.utils.dataset_utils import add_compute_stats
import pickle
from src.utils.misc import weblog_dataset_info
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import *
from src.utils.callbacks import *
import glob


def load_dataset(path):
    files = glob.glob(path + '/**')
    images = []
    labels = []
    for file in files:
        f = pickle.load(open(file, 'rb'))
        img, lb = f['images'], f['labels']
        images.extend(img)
        labels.extend(lb)
    return images, labels

def config_to_path_train(config):
    return f"coco_{config.network_name}"


class COCO_circle_size:
    def __init__(self, img, lb, indexes):
        self.images = img
        self.labels = lb
        self.indexes = indexes
        if not hasattr(self, 'transform'):
            self.transform = None

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]) if self.transform else self.images, \
                   torch.tensor(self.labels[idx])


config = Config(stop_when_train_acc_is=95,
                cuda_device_num=2,
                project_name='Ebbinghaus',
                list_tags=['learn_size', 'coco'],
                batch_size=64,
                network_name='resnet152',
                weblogger=2,  #set to "2" if you want to log into neptune client - if you do, you need to have an API token set (see neptune website). Otherwise put to 0.
                pretraining='vanilla',
                continue_train=False,  # watch out! With adam opt, setting this to True will not work as you intend!
                learning_rate=0.0001,
                img_size=75,
                clip_grad_norm=0.5,
                weight_decay=0.0,
                is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False)

torch.cuda.set_device(config.cuda_device_num) if torch.cuda.is_available() else None
config.net = ResNet152_size(imagenet_pt=True)

config.model_output_filename = './models/' + config_to_path_train(config) + '.pt'

if config.continue_train:
    config.pretraining = config.model_output_filename

# for param in config.net.net.parameters():
#     param.requires_grad = False

path = './data/coco_2017_size'

p = pickle.load(open(path + '_indexes.pickle', 'rb'))
train_idx, test_idx =p['train_idx'], p['test_idx']

images, labels = load_dataset(path)
config.loss_fn = torch.nn.MSELoss()
config.optimizer = torch.optim.Adam(config.net.parameters(),
                                    lr=config.learning_rate,
                                    weight_decay=config.weight_decay)

prepare_network(config.net, config, config.optimizer)


train_dataset = add_compute_stats(COCO_circle_size)(img=images, lb=labels, indexes=train_idx, name_ds='train', add_PIL_transforms=[RandomAffine(degrees=15, translate=(0.15, 0.15)), RandomHorizontalFlip(), CenterCrop((224, 224))], stats='ImageNet')



train_loader = DataLoader(train_dataset,
                           batch_size=config.batch_size,
                           drop_last=True,
                           sampler=SubsetRandomSampler(train_dataset.indexes),
                           num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                           timeout=0 if config.use_cuda and not config.is_pycharm else 0,
                           pin_memory=True)

weblog_dataset_info(train_loader, weblogger=config.weblogger, num_batches_to_log=1, log_text='train') if config.weblogger else None


test_dataset = add_compute_stats(COCO_circle_size)(img=images, lb=labels, indexes=test_idx, name_ds='test', add_PIL_transforms=[RandomAffine(degrees=0, translate=(0.15, 0.15)), CenterCrop((224, 224))], stats='ImageNet')


test_loaders = [DataLoader(test_dataset,
                           batch_size=config.batch_size,
                           drop_last=True,
                           sampler=SubsetRandomSampler(test_dataset.indexes),
                           num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                           timeout=0 if config.use_cuda and not config.is_pycharm else 0,
                           pin_memory=True)]

[weblog_dataset_info(td, weblogger=config.weblogger, num_batches_to_log=1, log_text='test') for td in test_loaders]

config.step = regression_step


def call_run(loader, train, callbacks, **kwargs):
    logs = {'ema_loss': ExpMovingAverage(0.2),
            'ema_mse': ExpMovingAverage(0.2),
            'ca_mse': CumulativeAverage(),
            'ema_rmse': ExpMovingAverage(0.2)}


    return run(loader,
               use_cuda=config.use_cuda,
               net=config.net,
               callbacks=callbacks,
               loss_fn=config.loss_fn,
               optimizer=config.optimizer,
               iteration_step=config.step,
               train=train,
               logs=logs,
               collect_data=kwargs['collect_data'] if 'collect_data' in kwargs else False,
               stats=train_dataset.stats)


def stop(logs, cb):
    logs['stop'] = True
    print('Early Stopping')

all_cb = [
    StopFromUserInput(),
    ProgressBar(l=len(train_dataset), batch_size=config.batch_size, logs_keys=['ema_loss', 'ema_mse', 'ema_rmse']),
    PrintNeptune(id='ema_loss', plot_every=10, weblogger=config.weblogger),
    PrintNeptune(id='ema_mse', plot_every=10, weblogger=config.weblogger),
    PrintNeptune(id='ema_rmse', plot_every=10, weblogger=config.weblogger),

    # Either train for X epochs
    # TriggerActionWhenReachingValue(mode='max', patience=1, value_to_reach=3500, check_every=10, metric_name='epoch', action=stop, action_name='10epochs'),
    #
    # Or explicitely traing until 9X% accuracy or convergence:
    TriggerActionWhenReachingValue(mode='min', patience=20, value_to_reach=0e-5, check_every=10, metric_name='ema_mse', action=stop, action_name=f'goal mse 0e-5'),

    # PlateauLossLrScheduler(config.optimizer, patience=1000, check_batch=True, loss_metric='ema_loss'),

    *[DuringTrainingTest(testing_loaders=test_ds, every_x_epochs=1, auto_increase=False, weblogger=config.weblogger, log_text='test during train', use_cuda=config.use_cuda, call_run=call_run, plot_samples_corr_incorr=True, callbacks=[
        PrintNeptune(id='ca_mse', plot_every=np.inf, log_prefix='test_EVAL&TRAIN_', weblogger=config.weblogger),
        PrintConsole(id='ca_mse', endln=" / ", plot_every=np.inf, plot_at_end=True),
        # PlotImagesEveryOnceInAWhile(config.weblogger, test_ds.dataset, plot_every=np.inf, plot_only_n_times=1, plot_at_the_end=True, max_images=10, text='')
                         ]) for test_ds in test_loaders]
]

# all_cb.append(ClipGradNorm(config.net, config.clip_grad_norm)) if config.clip_grad_norm is not None else None
all_cb.append(SaveModel(config.net, config.model_output_filename, optimizer=config.optimizer)) if not config.is_pycharm else None

net, logs = call_run(train_loader, True, all_cb)
config.weblogger.stop() if config.weblogger else None
