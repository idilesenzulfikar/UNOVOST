import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import numpy as np
from PIL import Image


class BBoxCrop(object):
    def __init__(self, padding=0):
        if type(padding) != int:
            raise TypeError('padding should be int')
        self.padding = padding

    def __call__(self, img, bbox):

        if not (isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox)) == 4:
            raise TypeError('bbox should be list or tuple or ndarray like [x,y,w,h]. Got {}'.format(type(bbox)))
        else:
            bbox = np.array(bbox).round().astype('int32')

        if not (isinstance(img, Image.Image)):
            raise TypeError('img should be PIL Image')
        else:
            width, height = img.size
            x0 = max(bbox[0] - self.padding, 0)
            y0 = max(bbox[1] - self.padding, 0)
            x1 = min((bbox[2] + self.padding), width)
            y1 = min((bbox[3] + self.padding), height)

            crop_img = img.crop((x0, y0, x1, y1))

        return crop_img


class Resize(object):
    def __init__(self, size):
        if not (isinstance(size, int) or (isinstance(size, (list, tuple, np.ndarray)) and len(size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(type(size)))

        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size[::-1]  # convert [w,h] to [h,w], check F.resize

    def __call__(self, img):
        img = F.resize(img, self.size, Image.BILINEAR)
        return img


def get_model(arch, device):
    if arch == 'vgg16':
        return models.vgg16(pretrained=True).to(device)
    elif arch == 'resnet101':
        return models.resnet101(pretrained=True).to(device)
    else:
        raise NotImplementedError


def extract_features(image, proposals, arch):
    bbox_crop = BBoxCrop()
    resize = Resize(224)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ReIDs = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(arch, device).eval()

    with torch.no_grad():
        for prop in proposals:
            bbox = prop['bbox']
            crop_img = bbox_crop(image, bbox)
            resized_img = resize(crop_img)
            batch = to_tensor(resized_img)
            batch = normalize(batch).unsqueeze(0).to('cuda')
            with torch.no_grad():
                feature = model(batch).cpu().numpy().squeeze()
            ReIDs.append(feature)

    return ReIDs
