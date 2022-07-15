import pickle
import sty
import os
import torch
import glob
import PIL.Image as Image
class RegressionDataset:
    def __init__(self, root, transform=None):
        """
        :param root: the path for the pickle file
        :param transform:
        """
        self.transform = transform
        files = glob.glob(root + '/*.png')
        self.images = []
        self.labels = []
        print(sty.fg.yellow + f"Loading dataset from {root} ...", end='' )
        for f in files:
            self.images.append(Image.open(f).convert('RGB'))
            self.labels.append(float(os.path.splitext(os.path.basename(f))[0]))

        print("Done." + sty.rs.fg)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return ((self.transform(self.images[idx]) if self.transform else self.images[idx]),
               torch.tensor(self.labels[idx]))