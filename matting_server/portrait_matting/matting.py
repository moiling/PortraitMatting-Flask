import torch
import numpy as np

from PIL import Image
from torchvision.transforms import functional as F

from portrait_matting.comp.estimate_fb import estimate_foreground_background
from portrait_matting import transforms
from portrait_matting.networks.matting_net import MattingNet


class Matting:
    def __init__(self, checkpoint_path='', gpu=False):
        torch.set_flush_denormal(True)  # flush cpu subnormal float.
        self.checkpoint_path = checkpoint_path
        self.gpu = gpu
        self.model = self.__load_model()

    def __load_model(self):
        model = MattingNet()
        if self.gpu and torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()

        # load checkpoint.
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def matting(self, image_numpy, return_img_trimap=False, img_size_in_net=-1, img_size_return=-1):
        with torch.no_grad():
            image = self.__load_image_tensor(image_numpy, img_size_return)
            if self.gpu and torch.cuda.is_available():
                image = image.cuda()
            else:
                image = image.cpu()

            b, c, h, w = image.shape

            # resize to training size.
            if img_size_in_net > 0:
                resize_image = F.resize(image, [img_size_in_net, img_size_in_net], Image.BILINEAR)
                pred_matte, pred_trimap_prob, _ = self.model(resize_image)
                pred_matte = F.resize(pred_matte, [h, w])
                pred_trimap_prob = F.resize(pred_trimap_prob, [h, w], Image.BILINEAR)
            else:
                pred_matte, pred_trimap_prob, _ = self.model(image)

            pred_matte = pred_matte.cpu().detach().squeeze(dim=0).numpy().transpose(1, 2, 0)
            image = image.cpu().detach().squeeze(dim=0).numpy().transpose(1, 2, 0)

            pred_trimap = pred_trimap_prob.squeeze(dim=0).softmax(dim=0).argmax(dim=0)
            pred_trimap = pred_trimap.cpu().detach().unsqueeze(dim=2).numpy() / 2.

            if not return_img_trimap:
                return pred_matte

            return pred_matte, image, pred_trimap

    @staticmethod
    def cutout(image, matte):
        fg, _ = estimate_foreground_background(image[..., ::-1], matte)  # [H, W, BGR(3) ]
        cutout = np.zeros((image.shape[0], image.shape[1], 4))
        cutout[..., :3] = fg[..., ::-1]
        cutout[...,  3] = matte.astype(np.float32).squeeze(axis=2)       # [H, W, RGBA(4)]
        return cutout

    @staticmethod
    def composite(cutout, bg):
        alpha = cutout[:, :, 3:4]
        fg    = cutout[:, :,  :3]
        comp = alpha * fg + (1 - alpha) * bg
        return comp

    def __load_image_tensor(self, image_numpy, max_size=-1):
        image = Image.fromarray(image_numpy[:, :, ::-1]).convert('RGB')
        if max_size > 0:
            [image] = transforms.ResizeIfBiggerThan(max_size)([image])
        [image] = transforms.ToTensor()([image])
        image = image.unsqueeze(dim=0)
        return image

    def __load_trimap_tensor(self, trimap_path, max_size=-1):
        if trimap_path is None:
            return None
        trimap = Image.open(trimap_path).convert('L')

        if max_size > 0:
            [trimap] = transforms.ResizeIfBiggerThan(max_size)([trimap])
        [trimap] = transforms.ToTensor()([trimap])

        # get 3-channels trimap.
        trimap_3 = trimap.repeat(3, 1, 1)
        trimap_3[0, :, :] = (trimap_3[0, :, :] <= 0.1).float()
        trimap_3[1, :, :] = ((trimap_3[1, :, :] < 0.9) & (trimap_3[1, :, :] > 0.1)).float()
        trimap_3[2, :, :] = (trimap_3[2, :, :] >= 0.9).float()

        trimap_3 = trimap_3.unsqueeze(dim=0)
        return trimap_3
