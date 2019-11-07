
from vision import VisionDataset

from PIL import Image

import os
import os.path
import sys
from model_vgg import VGG_16
import numpy as np
import torchvision.transforms as transforms
import torch

streams = 15
step = 3
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#prev = '/home/SharedData/Divakar/project1/curriculum/previous_inc.pth.tar'


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir,stream, device, prev, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    model = VGG_16(3,streams,step).to(device)
    model.load_state_dict(torch.load(prev))
    model.eval()
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for idx,target in (class_to_idx.items()):
        d = os.path.join(dir, str(int(target+1)))
        if not os.path.isdir(d):
            continue
        
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                #print('p',path)
                #print(class_to_idx)
                #print(len(class_to_idx))
                #print(target)
                if is_valid_file(path):

                    image = Image.open(path).resize((224,224))
                    image = np.array(image)
                    image = image.astype(np.float32)/255.0
                    #trans = transforms.ToPILImage()
                    trans1 = transforms.ToTensor()
                    image =  trans1((image))
                    image = image.unsqueeze(0)
                    with torch.no_grad():
                         image = image.to(device)
                         #print('malloc', torch.cuda.memory_allocated(device=device))
                         _,_,_,dist = model(image,stream)
                         #del(image)
                         
                    #dist = dist.to('cpu')
                    item = (path, class_to_idx[idx],dist)
                    images.append(item)

    #del(model) 
    return images


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root,stream, device, prev, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root,stream,device,prev, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, self.stream, self.device, self.prev,  class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        #print('1',classes)
        #print('2',len(classes))
        classes = [(int(i)-1) for i in classes]
        #print('3',classes)
        #print('4',len(classes))
        class_to_idx = {i:classes[i] for i in range(len(classes))}
        #print('5',class_to_idx)
        #print('6',len(class_to_idx))
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,dist = self.samples[index]
        #dist = dist.type(torch.LongTensor)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #print('typ',type(target))
        return sample, target, dist

    def __len__(self):
        #print('7',len(self.samples))
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder1(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, stream, device, prev, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolder1, self).__init__(root, stream, device, prev, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
