"""
This tests for the ebbinghaus illusion. Run it after having run "train".
The main things to change for other experiments are the testing datasets. Here I use EbbinghausTrain (I test on the train set, just to check all is ok), and the crucial EbbinghausTestSmallFlankers/EbbinghausTestBigFlankers
"""
import torch.nn
from src.ebbinghaus_illusion.ebbinghaus_datasets import EbbinghausTrain, EbbinghausTestBigFlankers, EbbinghausTestSmallFlankers
from src.utils.dataset_utils import add_compute_stats
from src.utils.net_utils import run, CumulativeAverage
from src.utils.Config import Config
from src.utils.net_utils import prepare_network
from torch.utils.data import DataLoader
from src.utils.callbacks import *
from src.decoder_method.train_utils import *
from src.utils.misc import weblog_dataset_info



config = Config(stop_when_train_acc_is=95,
                cuda_device_num=0,
                project_name='Ebbinghaus',
                batch_size=64,
                network_name='resnet152',
                weblogger=0,  # set to "2" if you want to log into neptune client - if you do, you need to have an API token set (see neptune website). Otherwise put to 0.
                pretraining='vanilla',
                continue_train=True,
                learning_rate=0.0001,
                clip_grad_norm=0.5,
                weight_decay=0.0,
                img_size=75,
                is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False)

torch.cuda.set_device(config.cuda_device_num) if torch.cuda.is_available() else None
config.net = ResNet152decoders(imagenet_pt=True)

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

[weblog_dataset_info(td, weblogger=config.weblogger, num_batches_to_log=1, log_text='test') for td in test_loaders]


def decoder_test(data, model, loss_fn, optimizer, use_cuda, logs, train, **kwargs):
    images, labels = data
    images = make_cuda(images, use_cuda)
    labels = make_cuda(labels, use_cuda)
    out_dec = model(images)
    [logs[f'ca_me_{i}'].add(torch.mean(out_dec[i]-labels).item()) for i in range(len(out_dec))]
    [logs[f'me_list_{i}'].extend((out_dec[i]-labels).tolist()) for i in range(len(out_dec))]
    return torch.tensor([0]), None, None, logs

config.step = decoder_test
import matplotlib.pyplot as plt
def call_run(loader, train, callbacks, **kwargs):
    logs = {f'ca_me_{i}': CumulativeAverage() for i in range(5)}
    logs.update({f'me_list_{i}': [] for i in range(5)})
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
    plt.figure()
    plt.title(loader.dataset.name_ds)
    [plt.hist(logs[f'me_list_{i}'], bins=20, alpha=0.6, label=i) for i in range(5)]
    plt.legend()
    plt.axvline(0, linestyle='--', color='r')
    plt.show()
    ##
    return [logs[f'me_list_{i}'] for i in range(5)]


def stop(logs, cb):
    logs['stop'] = True
    print('Early Stopping')


def gen_cb():
    return [
            StopFromUserInput(),
            ProgressBar(l=len(test_dataset_big), batch_size=config.batch_size, logs_keys=[f'ca_me_{i}' for i in range(5)]),
            StopWhenMetricIs(value_to_reach=0, metric_name='epoch', check_after_batch=False)
]

out_decoders = {tl.dataset.name_ds: call_run(tl, train=False, callbacks=gen_cb()) for tl in test_loaders}
config.weblogger.stop() if config.weblogger else None
out_decoders['test'][0]

from scipy.stats import ttest_ind

def compare_two_ds(n1, n2, decnum):
    plt.figure()
    stat = ttest_ind(out_decoders[n1][decnum], out_decoders[n2][decnum])
    if stat.pvalue < 0.0001:
        plt.xlabel(f"Dec {decnum}: diff. statistically significant: {stat.pvalue:.5f}!")
    else:
        plt.xlabel(f"Dec {decnum}: Diff. ~~~NOT~~~ statistically significant : {stat.pvalue:.5f}!")
    plt.hist(out_decoders[n1][decnum], bins=20, alpha=0.6, label=n1)
    plt.hist(out_decoders[n2][decnum], bins=20, alpha=0.6, label=n2)
    plt.legend()
    plt.title(f"Decoder {decnum}")
    plt.show()

compare_two_ds('test', 'test_big_fl', 0)
compare_two_ds('test', 'test_big_fl', 1)
compare_two_ds('test', 'test_big_fl', 2)
compare_two_ds('test', 'test_big_fl', 3)
compare_two_ds('test', 'test_big_fl', 4)


compare_two_ds('test', 'test_small_fl', 0)
compare_two_ds('test', 'test_small_fl', 1)
compare_two_ds('test', 'test_small_fl', 2)
compare_two_ds('test', 'test_small_fl', 3)
compare_two_ds('test', 'test_small_fl', 4)
