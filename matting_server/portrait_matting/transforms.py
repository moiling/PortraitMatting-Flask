import math
from torchvision.transforms import functional as F


class ResizeIfBiggerThan(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        for idx, image in enumerate(images):
            max_size = max(image.size)
            if max_size > self.size:
                rate = self.size / float(max_size)
                h, w = math.ceil(rate * image.size[0]), math.ceil(rate * image.size[1])
                images[idx] = F.resize(image, [w, h])
        return images


class ToTensor(object):
    def __call__(self, images):
        for idx, image in enumerate(images):
            # convert numpy and PIL to tensor.
            images[idx] = F.to_tensor(image)
        return images


class Normalize(object):
    # call after to_tensor
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for idx, image in enumerate(images):
            if image.shape[0] == 3:
                images[idx] = F.normalize(image, mean=self.mean, std=self.std)
        return images
