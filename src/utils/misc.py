import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
try:
    from neptune.new.types import File
    import neptune.new as neptune

except:
    pass

class ConfigSimple:
    def __init__(self, **kwargs):
        self.use_cuda = torch.cuda.is_available()
        self.verbose = True
        [self.__setattr__(k, v) for k, v in kwargs.items()]

    def __setattr__(self, *args, **kwargs):
        super().__setattr__(*args, **kwargs)




def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image


def convert_normalized_tensor_to_plottable_array(tensor, mean, std, text):
    image = conver_tensor_to_plot(tensor, mean, std)

    canvas_size = np.shape(image)

    font_scale = np.ceil(canvas_size[1])/150
    font = cv2.QT_FONT_NORMAL
    umat = cv2.UMat(image * 255)
    umat = cv2.putText(img=cv2.UMat(umat), text=text, org=(0, int(canvas_size[1] - 3)), fontFace=font, fontScale=font_scale, color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=6)
    umat = cv2.putText(img=cv2.UMat(umat), text=text, org=(0, int(canvas_size[1] - 3)),
                fontFace=font, fontScale=font_scale, color=[255, 255, 255], lineType=cv2.LINE_AA, thickness=1)
    image = cv2.UMat.get(umat)
    image = np.array(image, np.uint8)
    return image



def weblog_dataset_info(dataloader, log_text='', dataset_name=None, weblogger=1, plotter=None, num_batches_to_log=2):
    stats = {}
    if plotter is None:
        plotter = plot_images_on_weblogger
    if 'stats' in dir(dataloader.dataset):
        dataset = dataloader.dataset
        dataset_name = dataset.name_ds
        stats = dataloader.dataset.stats
    else:
        dataset_name = 'no_name' if dataset_name is None else dataset_name
        stats['mean'] = [0.5, 0.5, 0.5]
        stats['std'] = [0.2, 0.2, 0.2]
        Warning('MEAN, STD AND DATASET_NAME NOT SET FOR NEPTUNE LOGGING. This message is not referring to normalizing in PyTorch')

    if isinstance(weblogger, neptune.Run):
        weblogger['Logs'] = f'{dataset_name} mean: {stats["mean"]}, std: {stats["std"]}'

    for idx, data in enumerate(dataloader):
        plotter(dataset_name=dataset_name, data=data, stats=stats, weblogger=weblogger, text=log_text, batch_num=idx)
        if idx + 1 >= num_batches_to_log:
            break

def plot_images_on_weblogger(data, stats, weblogger=2, text='', **kwargs):# images, labels, more, log_text, weblogger=2):
    images, labels, *more = data
    plot_images = images[0:np.max((4, len(images)))]
    metric_str = 'Debug/{} example images'.format(text)
    lab = [f'{i.item():.3f}' for i in labels]
    # ax = imshow_batch(images, stats, lab)
    if isinstance(weblogger, neptune.Run):
        [weblogger[metric_str].log
                           (File.as_image(convert_normalized_tensor_to_plottable_array(im, stats['mean'], stats['std'], text=lb)/255))
         for im, lb in zip(plot_images, lab)]



def imshow_batch(inp, stats=None, labels=None, title_more='', maximize=True, ax=None):
    if stats is None:
        mean = np.array([0, 0, 0])
        std = np.array([1, 1, 1])
    else:
        mean = stats['mean']
        std = stats['std']
    """Imshow for Tensor."""

    cols =  int(np.ceil(np.sqrt(len(inp))))
    if ax is None:
        fig, ax = plt.subplots(cols, cols)
    if not isinstance(ax, np.ndarray):
        ax = np.array(ax)
    ax = ax.flatten()
    mng = plt.get_current_fig_manager()
    try:
        mng.window.showMaximized() if maximize else None
    except AttributeError:
        print("Tkinter can't maximize. Skipped")
    for idx, image in enumerate(inp):
        image = conver_tensor_to_plot(image, mean, std)
        ax[idx].clear()
        ax[idx].axis('off')
        if len(np.shape(image)) == 2:
            ax[idx].imshow(image, cmap='gray', vmin=0, vmax=1)
        else:
            ax[idx].imshow(image)
        if labels is not None and len(labels) > idx:
            if isinstance(labels[idx], torch.Tensor):
                t = labels[idx].item()
            else:
                t = labels[idx]
            text = str(labels[idx]) + ' ' + (title_more[idx] if title_more != '' else '')
            # ax[idx].set_title(text, size=5)
            ax[idx].text(0.5, 0.1, labels[idx], horizontalalignment='center', transform=ax[idx].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(top=1,
                        bottom=0.01,
                        left=0,
                        right=1,
                        hspace=0.01,
                        wspace=0.01)
    return ax


def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image
