import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

# Map labelId -> trainId. The Cityscapes dataset contains 34 labels, but they need to be remapped to 19 classes to train models.
# This mapping follows the standard approach suggested in the original cityscapes scripts at https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
# with the exception of the labels belonging to the 'void' category assigned to the 20th class for void classification
labelId_to_trainId = {
    0: 19,  1: 19,  2: 19,  3: 19,  4: 19,  5: 19,  6: 19,
    7: 0,   8: 1,   9: 255, 10: 255,
    11: 2,  12: 3,  13: 4,  14: 255, 15: 255, 16: 255,
    17: 5,  18: 255, 19: 6,  20: 7,
    21: 8,  22: 9,  23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15,
    29: 255,30: 255,31: 16, 32: 17,
    33: 18
}

def convert_labelId_to_trainId(label_img):
    """
    Convert labelId numpy array to trainId using the mapping defined above.
    """
    train_id_img = np.ones_like(label_img) * 255  # default ignore
    
    for label_id, train_id in labelId_to_trainId.items():
        train_id_img[label_img == label_id] = train_id
    
    return train_id_img

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)




class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        label_np = np.array(label)
        label_mapped = convert_labelId_to_trainId(label_np)
        label = Image.fromarray(label_mapped.astype(np.uint8))
    
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

