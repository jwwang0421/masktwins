import torch
import random
from mmseg.ops import resize


def build_mask_generator(cfg):
    if cfg is None:
        return None
    t = cfg.pop('type')
    if t == 'block':
        return BlockMaskGenerator(**cfg)
    else:
        raise NotImplementedError(t)


class BlockMaskGenerator:
    def __init__(self, mask_ratio=0.5, mask_block_size=64):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs,mask_ratio=0.5):
        # self.mask_block_size = imgs.shape[-1]/16
        if len(imgs.shape) == 4:
            B, _, H, W = imgs.shape

        elif len(imgs.shape) == 3:
            B, H, W = imgs.shape
        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)

        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > mask_ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        if len(imgs.shape) == 3:
            input_mask = input_mask.squeeze(1)
        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs,if2mask=False,ifCMask=False):
        # self.mask_block_size = imgs.shape[-1]/16
        mr=random.choice(self.mask_ratio)
        input_mask = self.generate_mask(imgs,mask_ratio=mr)
        if not if2mask:
            return imgs * input_mask
        elif ifCMask:
            imgs_m1=imgs * input_mask
            imgs_m2=imgs * (1.0-input_mask)
        else:
            input_mask2 = self.generate_mask(imgs,mask_ratio=1-mr)
            imgs_m1=imgs * input_mask
            imgs_m2=imgs * input_mask2
            
        return {'1':imgs_m1,'2':imgs_m2}
        
        

