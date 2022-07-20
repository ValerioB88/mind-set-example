"""
This tests for the ebbinghaus illusion. Run it after having run "train".
The main things to change for other experiments are the testing datasets. Here I use EbbinghausTrain (I test on the train set, just to check all is ok), and the crucial EbbinghausTestSmallFlankers/EbbinghausTestBigFlankers
"""

import torch.nn
from src.ebbinghaus.generate_datasets import EbbinghausRandomFlankers, EbbinghausTestBigFlankers, EbbinghausTestSmallFlankers
from src.utils.net_utils import run, CumulativeAverage, ExpMovingAverage
from src.utils.net_utils import prepare_network
from torch.utils.data import DataLoader
from src.utils.callbacks import *
from src.utils.decoder.train_utils import *
from src.utils.misc import weblog_dataset_info
from src.utils.decoder.data_utils import RegressionDataset
import argparse
import matplotlib.pyplot as plt
##### NEEEDS FIXING!!!!
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


parser = argparse.ArgumentParser()

parser.add_argument('--model_path')
parser.add_argument('--gpu_num', help='what gpu to use (int, starts at 0)?', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)


config.use_cuda = torch.cuda.is_available()
config.is_pycharm = True if 'PYCHARM_HOSTED' in os.environ else False
torch.cuda.set_device(config.gpu_num) if torch.cuda.is_available() else None

config.net = ResNet152decoders(imagenet_pt=True)
num_decoders = len(config.net.decoders)


config.pretraining = config.model_path

prepare_network(config.net, config, train=False)

test_dataset = fix_dataset(RegressionDataset('./data/ebbinghaus/test_random_data'), 'test')
test_dataset_small = fix_dataset(RegressionDataset('./data/ebbinghaus/test_small_flankers_data'), 'test_small')
test_dataset_big = fix_dataset(RegressionDataset('./data/ebbinghaus/test_big_flankers_data'), 'test_big')

num_decoders = len(config.net.decoders)

#
test_datasets = [test_dataset, test_dataset_big, test_dataset_small]

test_loaders = [DataLoader(td,
                           batch_size=config.batch_size,
                           drop_last=True,
                           num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                           timeout=0 if config.use_cuda and not config.is_pycharm else 0,
                           pin_memory=True) for td in test_datasets]

# [weblog_dataset_info(td, weblogger=config.weblogger, num_batches_to_log=1, log_text=f'test {td.dataset.name_ds}') for td in test_loaders]

config.loss_fn = torch.nn.MSELoss()


def decoder_test(data, model, loss_fn, optimizer, use_cuda, logs, train, **kwargs):
    images, labels = data
    images = make_cuda(images, use_cuda)
    labels = make_cuda(labels, use_cuda)
    # imshow_batch(images.cpu(), stats=kwargs['stats'], labels=[i.item() for i in labels])
    # plt.show()
    with torch.no_grad():

        out_dec = model(images)
        loss = make_cuda(torch.tensor([0.]), use_cuda)
        mse_dec = []
        for idx, od in enumerate(out_dec):
            mse_dec.append(loss_fn(od,
                labels))
            loss += mse_dec[-1]
        logs[f'ca_rmse'].add(torch.sqrt(loss/num_decoders).item())
        [logs[f'ema_rmse_{idx}'].add(torch.sqrt(ms)) for idx, ms in enumerate(mse_dec)]

        # [logs[f'ca_rmse_{i}'].add(torch.sqrt(torch.mean(torch.square(out_dec[i]-labels))).item()) for i in range(len(out_dec))]
        # [logs[f'me_list_{i}'].extend((out_dec[i]).tolist()) for i in range(len(out_dec))]
        [logs[f'me_list_{i}'].extend((out_dec[i]-labels).tolist()) for i in range(len(out_dec))]

    return torch.tensor([0]), None, None, logs

config.step = decoder_test

def call_run(loader, train, callbacks, **kwargs):
    num_decoders = 6
    # logs = {f'ca_me_{i}': CumulativeAverage() for i in range(num_decoders)}
    logs = {}
    logs.update({f'ema_rmse_{i}': ExpMovingAverage(0.2) for i in range(6)})
    logs.update({'ca_rmse': CumulativeAverage()})

    logs.update({f'me_list_{i}': [] for i in range(num_decoders)})
    print(sty.fg.red + sty.ef.inverse + f"***** DATASET {loader.dataset.name_ds} *****" + sty.rs.fg + sty.rs.ef)
    _, logs = run(loader,
               use_cuda=config.use_cuda,
               net=config.net,
               callbacks=callbacks,
               loss_fn=config.loss_fn,
               optimizer=None,
               iteration_step=config.step,
               train=train,
               logs=logs,
               collect_data=kwargs['collect_data'] if 'collect_data' in kwargs else False,
               stats=test_dataset_big.stats)

    # plt.figure()
    # plt.title(loader.dataset.name_ds)
    # [plt.hist(logs[f'me_list_{i}'], bins=20, alpha=0.6, label=i) for i in range(num_decoders)]
    # plt.legend()
    # plt.axvline(0, linestyle='--', color='r')
    # plt.show()
    return [logs[f'me_list_{i}'] for i in range(num_decoders)]


def stop(logs, cb):
    logs['stop'] = True
    print('Early Stopping')


def gen_cb():
    return [
            StopFromUserInput(),
            ProgressBar(l=len(test_dataset_big), batch_size=config.batch_size, logs_keys=[f'ema_rmse_{i}' for i in range(num_decoders)]),
            StopWhenMetricIs(value_to_reach=0, metric_name='epoch', check_after_batch=False)
]

out_decoders = {tl.dataset.name_ds: call_run(tl, train=False, callbacks=gen_cb()) for tl in test_loaders}
config.weblogger.stop() if config.weblogger else None


def compare_two_ds(decnum, *args):
    color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # plt.figure()
    # stat = ttest_ind(out_decoders[n1][decnum], out_decoders[n2][decnum])
    # if stat.pvalue < 0.0001:
    #     plt.xlabel(f"Dec {decnum}: diff. statistically significant: {stat.pvalue:.5f}!")
    # else:
    #     plt.xlabel(f"Dec {decnum}: Diff. ~~~NOT~~~ statistically significant : {stat.pvalue:.5f}!")
    for idx, n in enumerate(args):
        plt.hist(out_decoders[n][decnum], bins=20, density=True, alpha=0.6, label=n, color=color_cycle[idx])
        plt.axvline(np.median(out_decoders[n][decnum]), linestyle='--', color=color_cycle[idx])


    plt.legend()
    plt.title(f"Decoder {decnum}")
    plt.show()


compare_two_ds(0, 'test', 'small', 'big')
compare_two_ds(1, 'test', 'small', 'big')
compare_two_ds(2, 'test', 'small', 'big')
compare_two_ds(3, 'test', 'small', 'big')
compare_two_ds(4, 'test', 'small', 'big')
compare_two_ds(5, 'test', 'small', 'big')