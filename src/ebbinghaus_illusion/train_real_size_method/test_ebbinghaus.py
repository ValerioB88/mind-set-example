import torch.nn
import torchvision
from src.ebbinghaus_illusion.ebbinghaus_datasets import *
from src.ebbinghaus_illusion.train_real_size_method.train_utils import ResNet152_size
from src.utils.Config import Config
from src.utils.dataset_utils import add_compute_stats
from src.utils.net_utils import prepare_network, CumulativeAverage, run, make_cuda
from torch.utils.data import DataLoader

# from src.decoder_method.dataset_utils import *
from src.utils.misc import weblog_dataset_info
from src.utils.callbacks import *

def config_to_path_train(config):
    return f"{config.network_name}"


config = Config(
                cuda_device_num=3,
                list_tags = ['coco', 'learn_size', 'test_ebbingh'],
                project_name='Ebbinghaus',
                batch_size=64,
                network_name='resnet152',
                weblogger=0,  # set to "2" if you want to log into neptune client - if you do, you need to have an API token set (see neptune website). Otherwise put to 0.
                pretraining='./models/coco_resnet152.pt',
                continue_train=False,
                img_size=75,
                is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False)

torch.cuda.set_device(config.cuda_device_num) if torch.cuda.is_available() else None
config.net = ResNet152_size(imagenet_pt=True)

config.model_output_filename = './models/' + config_to_path_train(config) + '.pt'

if config.continue_train:
    config.pretraining = config.model_output_filename

prepare_network(config.net, config)
for param in config.net.net.parameters():
    param.requires_grad = False


test_dataset = add_compute_stats(EbbinghausTrain)(name_ds='test', add_PIL_transforms=[torchvision.transforms.Resize(224)], stats='ImageNet', size_dataset=200, img_size=config.img_size, background='black')
test_dataset_small = add_compute_stats(EbbinghausTestSmallFlankers)(name_ds='test_small_fl', add_PIL_transforms=[torchvision.transforms.Resize(224)], stats='ImageNet', size_dataset=200, img_size=config.img_size, background='black')
test_dataset_big = add_compute_stats(EbbinghausTestBigFlankers)(name_ds='test_big_fl', add_PIL_transforms=[torchvision.transforms.Resize(224)], stats='ImageNet', size_dataset=200, img_size=config.img_size, background='black')

test_datasets = [test_dataset, test_dataset_big, test_dataset_small]
test_loaders = [DataLoader(td,
                           batch_size=config.batch_size,
                           drop_last=True,
                           num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                           timeout=0 if config.use_cuda and not config.is_pycharm else 0,
                           pin_memory=True) for td in test_datasets]

[weblog_dataset_info(td, weblogger=config.weblogger, num_batches_to_log=1, log_text=td.dataset.name_ds) for td in test_loaders]


def decoder_test(data, model, loss_fn, optimizer, use_cuda, logs, train, **kwargs):
    images, labels = data
    images = make_cuda(images, use_cuda)
    labels = make_cuda(labels, use_cuda)
    out = model(images)
    logs[f'ca_me'].add(torch.mean(out.squeeze()-labels).item())
    logs[f'size_list'].extend((out.squeeze()).tolist())
    logs['true_list'].extend(labels.tolist())
    return torch.tensor([0]), None, None, logs

config.step = decoder_test
import matplotlib.pyplot as plt

def call_run(loader, train, callbacks, **kwargs):
    logs = {f'ca_me': CumulativeAverage(), f'size_list': [], 'true_list': []}
    print(sty.fg.red + sty.ef.inverse + f"***** DATASET {loader.dataset.name_ds} *****" + sty.rs.fg + sty.rs.ef)
    _, logs= run(loader,
               use_cuda=config.use_cuda,
               net=config.net,
               callbacks=callbacks,
               loss_fn=None,
               optimizer=None,
               iteration_step=config.step,
               train=train,
               logs=logs,
               collect_data=kwargs['collect_data'] if 'collect_data' in kwargs else False,
               stats=test_dataset.stats)

    ##
    # plt.figure()
    # plt.title(loader.dataset.name_ds)
    # plt.hist(logs[f'size_list'], alpha=0.6)
    # plt.legend()
    # plt.axvline(0, linestyle='--', color='r')
    # plt.show()
    ##
    return logs


def stop(logs, cb):
    logs['stop'] = True
    print('Early Stopping')


def gen_cb():
    return  [
    StopFromUserInput(),
    ProgressBar(l=len(test_dataset_big), batch_size=config.batch_size, logs_keys=[f'ca_me' for i in range(5)]),
    StopWhenMetricIs(value_to_reach=0, metric_name='epoch', check_after_batch=False)
]



logs = {tl.dataset.name_ds: call_run(tl, train=False, callbacks=gen_cb()) for tl in test_loaders}

from scipy.stats import ttest_ind

def compare_two_ds(n1, n2):
    plt.figure(1)

    stat = ttest_ind(logs[n1]['size_list'], logs[n2]['size_list'])
    if stat.pvalue < 0.0001:
        plt.xlabel(f"Diff. statistically significant: {stat.pvalue:.5f}!")
    else:
        plt.xlabel(f"Diff. ~~~NOT~~~ statistically significant : {stat.pvalue:.5f}!")
    plt.hist(logs[n1]['size_list'], bins=20, alpha=0.6, label=n1)
    plt.hist(logs[n2]['size_list'], bins=20, alpha=0.6, label=n2)
    plt.legend()
    plt.show()
    plt.figure(2)
    plt.scatter(logs[n1]['true_list'], logs[n1]['size_list'])
    plt.scatter(logs[n2]['true_list'], logs[n2]['size_list'])
    plt.show()

compare_two_ds('test_small_fl', 'test_big_fl')
# compare_two_ds('test', 'test_small_fl')

config.weblogger.stop() if config.weblogger else None

##

