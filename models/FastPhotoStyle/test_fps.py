import torch 
import torch.nn as nn 
import numpy as np
import time
import torchvision.utils as utils
from PIL import Image
from StyleTransfer.models import get_model

class Timer:
    def __init__(self, msg, logger):
        self.msg = msg
        self.start_time = None
        self.logger = logger

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.logger:
            self.logger.info(self.msg % (time.time() - self.start_time))
        else:
            print(self.msg % (time.time() - self.start_time))

def prepare_fps_model(cfg):
    # create model 
    model = get_model(cfg.MODEL.NAME, cfg)
    # load model weights
    model.load(cfg)
    # push model to device
    model.to(cfg.DEVICE)
    return model

def fps_style_transfer(model, content_img, style_img, ch, cw, content_seg=[], style_seg=[], logger=None, orig_content=None, test_transform=None):
    # load images and takes only one channel
    # all channels are actually same to each other
    content_seg = np.asarray(content_seg)
    style_seg = np.asarray(style_seg)

    if content_seg.size > 0:
        content_seg = content_seg[:, :, 0]
    if style_seg.size > 0:
        style_seg = style_seg[:, :, 0]

    with Timer('Elapsed time in stylization: %f', logger):
        with torch.no_grad():
            stylized_img = model.transform(content_img, style_img, content_seg, style_seg)

    _, _, new_ch, new_cw = stylized_img.shape 
    if ch != new_ch or cw != new_cw:
        if logger:
            logger.info('De-resize image: (%d, %d) -> (%d, %d)' % (new_ch, new_cw, ch, cw))
        else:
            print('De-resize image: (%d, %d) -> (%d, %d)' %
                  (new_ch, new_cw, ch, cw))
        stylized_img = torch.nn.functional.interpolate(stylized_img, size=(ch, cw), mode='bilinear', align_corners=False)
    
    grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    out_img = Image.fromarray(ndarr)

    with Timer('Elapsed time in propagation: %f\n', logger):
        assert orig_content is not None, 'Smoothing step in FastPhotoStyle requires original content image'
        out_img = model.smooth(out_img, orig_content)
    
    return out_img  # PIL image


def fps_style_transfer_non_photo(model, content_img, style_img, ch, cw, alpha=1.0, logger=None, style_interp_weights=[], mask_img=None):
    if mask_img:
        return None  # support mask
    elif style_interp_weights:
        return None  # support interpolation
    else:
        # load images and takes only one channel
        # all channels are actually same to each other
        content_seg = np.asarray([])
        style_seg = np.asarray([])

        if content_seg.size > 0:
            content_seg = content_seg[:, :, 0]
        if style_seg.size > 0:
            style_seg = style_seg[:, :, 0]

        with Timer('Elapsed time in stylization: %f', logger):
            with torch.no_grad():
                stylized_img = model.transform(content_img, style_img, content_seg, style_seg)

        _, _, new_ch, new_cw = stylized_img.shape 
        if ch != new_ch or cw != new_cw:
            if logger:
                logger.info('De-resize image: (%d, %d) -> (%d, %d)' % (new_ch, new_cw, ch, cw))
            else:
                print('De-resize image: (%d, %d) -> (%d, %d)' %
                    (new_ch, new_cw, ch, cw))
            stylized_img = torch.nn.functional.interpolate(stylized_img, size=(ch, cw), mode='bilinear', align_corners=False)
    
        grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray(ndarr)

        return out_img  # PIL image
