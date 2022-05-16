from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from tqdm import tqdm
import pickle
import pathlib
import matplotlib.pyplot as plt
import sty

folder = './data/coco_2017'
type = 'train'
annFile = f'{folder}/raw/instances_{type}2017.json'
coco = COCO(annFile)

folder = './data/coco_2017_size/'
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

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

def get_image_from_id(id, cat_id):
    size_crop = (300, 300)
    img = coco.loadImgs(id)[0]
    cropped_imgs = []
    I = io.imread(img['coco_url'])
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cat_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    ann_plot = []
    all_objs = []
    size = (img['width'], img['height'])
    eps = np.min((size[0] * 0.1, size[1] * 0.1))

    # print(len(anns))
    for i in range(len(anns) - 1, -1, -1):
        an = anns[i]
        ann_plot.append(an)
        box = an['bbox']
        if box[0] < 0 + eps or box[1] < 0 + eps or box[0] + box[2] > size[0] - eps or box[1] + box[3] > size[1] - eps:
            # print("Removed")
            anns.remove(an)

    for rnd_obj in anns:
        pil_img = Image.fromarray(I).convert("RGB")

        ## How much can we crop? It depends on the pixel size of bbox / image pixel size
        center_bbox = rnd_obj['bbox'][0] + rnd_obj['bbox'][2] // 2, rnd_obj['bbox'][1] + rnd_obj['bbox'][3] // 2
        ss = rnd_obj['bbox'][2] * rnd_obj['bbox'][3] / (img['height'] * img['width'])
        # px = 50 * img['width'] * ss, 50 * img['width'] * ss #20*img['height'] * ss
        left = center_bbox[0] - size_crop[0] // 2
        top = center_bbox[1] - size_crop[1] // 2
        right = center_bbox[0] + size_crop[0] // 2
        bottom = center_bbox[1] + size_crop[1] // 2

        crop_pil = pil_img.crop((left, top, right, bottom))
        cropped_imgs.append(crop_pil)
        all_objs.append(rnd_obj)
    return cropped_imgs, all_objs


def add_circle(img, bbox, radius):
    draw = ImageDraw.Draw(img)
    circle(draw, (np.random.uniform(img.size[0] // 2 - bbox[2] // 2, img.size[0] // 2 + bbox[2] // 2),
                  np.random.uniform(img.size[1] // 2 - bbox[3] // 2, img.size[1] // 2 + bbox[3] // 2)),
           radius=radius, fill=(255, 0, 0))


def circle(draw, center, radius, fill=None):
      draw.ellipse((center[0] - radius + 1,
                    center[1] - radius + 1,
                    center[0] + radius - 1,
                    center[1] + radius - 1), fill=fill, outline=None)


def main():
    all_labels = list(size_cm.keys())
    tot = 0
    for label in all_labels:
        catIds = coco.getCatIds(catNms=[label])
        imgIds = coco.getImgIds(catIds=catIds)
        tot += len(imgIds)
        print(f'{label} size: {len(imgIds)}')

    print(f"tot: {tot}")

    max_images_per_class = 6000

    ## get all images containing given categories, select one at random
    all_pil_images = []
    all_sizes = []

    for label in tqdm(all_labels):
        print(sty.fg.red + f"Label {label}" + sty.rs.fg)
        pbar = tqdm(total=max_images_per_class)

        # label = all_labels[1]
        catIds = coco.getCatIds(catNms=[label])
        imgIds = coco.getImgIds(catIds=catIds)
        counter = 0
        for i in tqdm(imgIds):
            try:
                cropped_imgs, ann_objs = get_image_from_id(i, catIds)
                # even better would be to place the circle anywhere within the segmented area, not the bbox. But it's ok for now
                if not cropped_imgs:
                    continue
                for img, ann in zip(cropped_imgs, ann_objs):
                    rd = np.random.uniform(5, 40)
                    add_circle(img, ann['bbox'], rd)
                    size = int((rd*2) / np.mean([ann['bbox'][2], ann['bbox'][3]]) * size_cm[label])

                    all_pil_images.append(img)
                    all_sizes.append(size)
                    pbar.update(1)
                    counter += 1
                    if counter > max_images_per_class:
                        break
            except:
                pass
            if counter > max_images_per_class:
                break

        pickle.dump({'images': all_pil_images, 'labels': all_sizes}, open(folder + f'{label}.pickle', 'wb'))


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

from sklearn.model_selection import train_test_split
import glob
import pickle

if __name__ == '__main__':
    main()

    path = './data/coco_2017_size'
    images, labels = load_dataset(path)
    train_idx, test_idx = train_test_split(range(len(images)), test_size=.05)
    pickle.dump({'train_idx': train_idx, 'test_idx': test_idx}, open('./data/coco_2017_size_indexes.pickle', 'wb'))

