import torch.nn
from pycocotools.coco import COCO
from src.ebbinghaus_illusion.train_real_size_method.train_utils import ResNet152_size
from src.utils.Config import Config
from src.utils.dataset_utils import add_compute_stats
from src.utils.net_utils import prepare_network, CumulativeAverage, run, make_cuda
from torch.utils.data import DataLoader
import torchvision
from src.utils.misc import weblog_dataset_info
import glob
from src.ebbinghaus_illusion.train_real_size_method.generate_size_dataset import get_image_from_id, circle
from PIL import ImageDraw
from src.utils.callbacks import *
import pickle

size_cm = {
           'tennis racket': 25,
           'person': 180,
           'traffic light': 75,
           'fire hydrant': 83,
           'stop sign': 76,
           'bottle': 15,
           'wine glass': 11,
           'orange': 8,
           'mouse': 12}
labels = list(size_cm.keys())

def config_to_path_train(config):
    return f"{config.network_name}"

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

class COCO_circle_size:
    def __init__(self):
        folder = './data/coco_2017'
        type = 'train'
        self.images = []
        self.labels = []

        annFile = f'{folder}/raw/instances_{type}2017.json'
        coco = COCO(annFile)
        label = labels[np.random.choice(range(len(labels)))]
        for _ in range(1000):
            catIds = coco.getCatIds(catNms=[label])
            imgIds = coco.getImgIds(catIds=catIds)
            id = imgIds[np.random.choice(range(len(imgIds)))]
            cropped_imgs, ann_objs = get_image_from_id(id, catIds)
            if not cropped_imgs:
                continue
            idx = np.random.choice(range(len(cropped_imgs)))
            img = cropped_imgs[idx]
            ann = ann_objs[idx]
            for rd in np.arange(5, 100, 0.5):
                new_img = img.copy()
                # add_circle(new_img, ann['bbox'], rd)
                draw = ImageDraw.Draw(new_img)
                circle(draw, (img.size[0] // 2, img.size[1] // 2),
                       radius=rd, fill=(255, 0, 0))

                size = int(((rd) / np.mean([ann['bbox'][2], ann['bbox'][3]])) * size_cm[label])
                self.images.append(new_img)
                self.labels.append(size)
            if len(self.images) > 0:
                break


        if not hasattr(self, 'transform'):
            self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]) if self.transform else self.images, \
                   torch.tensor(self.labels[idx])


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

test_dataset = add_compute_stats(COCO_circle_size)(name_ds='test', add_PIL_transforms=[torchvision.transforms.Resize(224)], stats='ImageNet')

test_datasets = [test_dataset]
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
    logs[f'pred_size'].extend((out.squeeze()).tolist())
    logs['true_size'].extend(labels.tolist())
    return torch.tensor([0]), None, None, logs

config.step = decoder_test
import matplotlib.pyplot as plt

def call_run(loader, train, callbacks, **kwargs):
    logs = {f'ca_me': CumulativeAverage(), f'pred_size': [], 'true_size': []}
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
    ProgressBar(l=len(test_dataset[0]), batch_size=config.batch_size, logs_keys=[f'ca_me' for i in range(5)]),
    StopWhenMetricIs(value_to_reach=0, metric_name='epoch', check_after_batch=False)
]



logs = {tl.dataset.name_ds: call_run(tl, train=False, callbacks=gen_cb()) for tl in test_loaders}

#
# def compare_two_ds(n1, n2):
#     plt.figure(1)
#
#     stat = ttest_ind(logs[n1]['size_list'], logs[n2]['size_list'])
#     if stat.pvalue < 0.0001:
#         plt.xlabel(f"Diff. statistically significant: {stat.pvalue:.5f}!")
#     else:
#         plt.xlabel(f"Diff. ~~~NOT~~~ statistically significant : {stat.pvalue:.5f}!")
#     plt.hist(logs[n1]['size_list'], bins=20, alpha=0.6, label=n1)
#     plt.hist(logs[n2]['size_list'], bins=20, alpha=0.6, label=n2)
#     plt.legend()
#     plt.show()
#     plt.figure(2)
#     plt.scatter(logs[n1]['true_size'], logs[n1]['pred_size'])
#     plt.scatter(logs[n2]['true_size'], logs[n2]['pred_size'])
#     plt.show()
#
# compare_two_ds('test_small_fl', 'test_big_fl')
# compare_two_ds('test', 'test_small_fl')


plt.scatter(logs['test']['true_size'], logs['test']['pred_size'])
plt.show()
i=0; plt.imshow(test_dataset.images[i]); plt.title(f'G: {logs["test"]["true_size"][i]} P:{logs["test"]["pred_size"][i]}'); plt.show()
config.weblogger.stop() if config.weblogger else None

logs['test']['ca_me'].value


##

