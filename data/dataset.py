from PIL import Image, ImageFile 
import torchvision.transforms as transforms
import torch.utils.data as data 
import os 
Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_img_file(filename):
    return any(filename.lower().endswith(extension) for extension in ('jpg', 'png', 'jpeg'))

def default_loader(path):
    return Image.open(path).convert('RGB')


class Resize(object):
    def __init__(self, fine_size, is_test=False, interpolation=Image.BICUBIC):
        super(Resize, self).__init__()
        self.fine_size = fine_size
        self.is_test = is_test
        self.interpolation = interpolation
    
    def __call__(self, image):
        w, h = image.size 
        if w < h:
            scaled_w = self.fine_size if not self.is_test else w // 8 * 8 
            scaled_h = (h / w) * scaled_w 
            scaled_h = scaled_h // 8 * 8 
        else:
            scaled_h = self.fine_size if not self.is_test else h // 8 * 8
            scaled_w = (w / h) * scaled_h 
            scaled_w = scaled_w // 8 * 8
        scaled_h, scaled_w = int(scaled_h), int(scaled_w)
        scaled_image = transforms.functional.resize(image, (scaled_h, scaled_w), interpolation=self.interpolation)
        return scaled_image


def build_transform(cfg, train=False, interpolation=Image.BICUBIC, normalize=True):
    if train:
        if normalize:
            t = transforms.Compose([
                # transforms.Resize(cfg.INPUT.FINE_SIZE),
                Resize(cfg.INPUT.FINE_SIZE, is_test=False, interpolation=interpolation),
                transforms.RandomCrop(cfg.INPUT.FINE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            t = transforms.Compose([
                # transforms.Resize(cfg.INPUT.FINE_SIZE),
                Resize(cfg.INPUT.FINE_SIZE, is_test=False, interpolation=interpolation),
                transforms.RandomCrop(cfg.INPUT.FINE_SIZE),
                transforms.RandomHorizontalFlip(),
            ])

    else:
        if normalize:
            t = transforms.Compose([
                # transforms.Resize(cfg.INPUT.FINE_SIZE),
                Resize(cfg.INPUT.FINE_SIZE, is_test=True, interpolation=interpolation),
                transforms.ToTensor()
            ])
        else:
            t = transforms.Compose([
                # transforms.Resize(cfg.INPUT.FINE_SIZE),
                Resize(cfg.INPUT.FINE_SIZE, is_test=True, interpolation=interpolation),
            ])

    return t


"""
WCT, FastPhotoStyle - learning-free
AdaIN - learning required

For learning-free methods, we do inference only, i.e., the desired content and style
images should be given as a pair. 

For learning-required methods, we should pair content images with different kinds of 
style images during training. It is done by building separate datasets for content
and style images - DatasetNoSeg. However, in test mode, the content and style images 
should be paired.
"""

class DatasetNoSeg(data.Dataset):
    def __init__(self, cfg, directory, train=True):
        super(DatasetNoSeg, self).__init__()
        self.img_list = [x for x in os.listdir(directory) if is_img_file(x)]
        self.img_list = sorted(self.img_list)
        self.dir = directory

        # build transform 
        self.transform = build_transform(cfg, train=train)
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.img_list[index])

        try:
            img = default_loader(img_path)
        except OSError as e:
            print(e)
            print(img_path)
            exit()

        img = self.transform(img)

        return img

