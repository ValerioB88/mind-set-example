"""
General training script for decoder approach. The only thing you need to change is the loading `DATASET` variable. Note that in this case the EbbinghausTrain dataset is always generated on the fly (but you could specify the kwarg "path" to save/load it on disk).
"""
from src.ebbinghaus.ebbinghaus_datasets import EbbinghausRandomFlankers
from src.utils.Config import Config
from src.ebbinghaus.decoder.train_utils import decoder_step, ResNet152decoders, config_to_path_train
from src.utils.dataset_utils import add_compute_stats
from src.utils.net_utils import prepare_network, ExpMovingAverage, CumulativeAverage, run
from torch.utils.data import DataLoader
from src.utils.misc import weblog_dataset_info
from src.utils.callbacks import *
import torchvision



config = Config(stop_when_train_acc_is=95,
                cuda_device_num=0,
                project_name='Ebbinghaus',
                batch_size=64,
                network_name='resnet152',
                weblogger=False,  #set to True if you want to log into neptune client - if you do, you need to have an API token set (see neptune website). Otherwise put to 0.
                pretraining='vanilla',
                continue_train=False,  # watch out! With adam opt, setting this to True will not work as you intend!
                learning_rate=0.000001,
                img_size=75,
                clip_grad_norm=0.5,
                weight_decay=0.0,
                is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False)

torch.cuda.set_device(config.cuda_device_num) if torch.cuda.is_available() else None
config.net = ResNet152decoders(imagenet_pt=True)

config.model_output_filename = './models/' + config_to_path_train(config) + '.pt'

if config.continue_train:
    config.pretraining = config.model_output_filename

for param in config.net.net.parameters():
    print(sty.fg.inverse + "FREEZING CORE NETWORK" + sty.rs.ef)
    param.requires_grad = False

config.loss_fn = torch.nn.MSELoss()
config.optimizer = torch.optim.Adam(config.net.parameters(),
                                    lr=config.learning_rate,
                                    weight_decay=config.weight_decay)

prepare_network(config.net, config, config.optimizer)

train_dataset = EbbinghausRandomFlankers(size_dataset=9800, img_size=config.img_size, background='black')
train_dataset.name_ds = 'train'
train_dataset.stats = {'mean': [0.491, 0.482, 0.44], 'std': [0.247, 0.243, 0.262]} # ImageNet stats
train_dataset.transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=train_dataset.stats['mean'],
                                                         std=train_dataset.stats['std'])])


train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          drop_last=True,
                          shuffle=True,
                          num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                          timeout=0 if config.use_cuda and not config.is_pycharm else 0,
                          pin_memory=True)

weblog_dataset_info(train_loader, weblogger=config.weblogger, num_batches_to_log=1, log_text='train') if config.weblogger else None


test_dataset = EbbinghausRandomFlankers(size_dataset=200, img_size=config.img_size, background='black')
test_dataset.name_ds = 'test'
test_dataset.stats = {'mean': [0.491, 0.482, 0.44], 'std': [0.247, 0.243, 0.262]}
test_dataset.transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=test_dataset.stats['mean'],
                                                         std=test_dataset.stats['std'])])


test_datasets = [test_dataset]
test_loaders = [DataLoader(td,
                           batch_size=config.batch_size,
                           drop_last=True,
                           num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                           timeout=0 if config.use_cuda and not config.is_pycharm else 0,
                           pin_memory=True) for td in test_datasets]

[weblog_dataset_info(td, weblogger=config.weblogger, num_batches_to_log=1, log_text='test') for td in test_loaders]

config.step = decoder_step


def call_run(loader, train, callbacks, **kwargs):
    logs = {'ema_loss': ExpMovingAverage(0.2),
            'ema_mse': ExpMovingAverage(0.2),
            'ca_mse': CumulativeAverage()}

    logs.update({f'ema_rmse_{i}': ExpMovingAverage(0.2) for i in range(5)})

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
    ProgressBar(l=len(train_dataset), batch_size=config.batch_size, logs_keys=['ema_loss', 'ema_mse', *[f'ema_rmse_{i}' for i in range(5)]]),
    PrintNeptune(id='ema_loss', plot_every=10, weblogger=config.weblogger),
    PrintNeptune(id='ema_mse', plot_every=10, weblogger=config.weblogger),
    # Either train for X epochs
    TriggerActionWhenReachingValue(mode='max', patience=1, value_to_reach=3500, check_every=10, metric_name='epoch', action=stop, action_name='10epochs'),

    # Or train until 9X% accuracy or convergence:
    TriggerActionWhenReachingValue(mode='min', patience=20, value_to_reach=0e-5, check_every=10, metric_name='ema_mse', action=stop, action_name=f'goal mse 0e-5'),


    *[DuringTrainingTest(testing_loaders=test_ds, every_x_epochs=1, auto_increase=False, weblogger=config.weblogger, log_text='test during train', use_cuda=config.use_cuda, call_run=call_run, plot_samples_corr_incorr=True, callbacks=[
        # if you don't use neptune, this will be ignored
        PrintNeptune(id='ca_mse', plot_every=np.inf, log_prefix='test_EVAL&TRAIN_', weblogger=config.weblogger),
        PrintConsole(id='ca_mse', endln=" / ", plot_every=np.inf, plot_at_end=True),
        # PlotImagesEveryOnceInAWhile(config.weblogger, test_ds.dataset, plot_every=np.inf, plot_only_n_times=1, plot_at_the_end=True, max_images=10, text='')
                         ]) for test_ds in test_loaders]
]

# all_cb.append(ClipGradNorm(config.net, config.clip_grad_norm)) if config.clip_grad_norm is not None else None
all_cb.append(SaveModel(config.net, config.model_output_filename, optimizer=config.optimizer)) if not config.is_pycharm else None

net, logs = call_run(train_loader, True, all_cb)
config.weblogger.stop() if config.weblogger else None
