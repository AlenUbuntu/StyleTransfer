from StyleTransfer.models import get_model
from StyleTransfer.config import get_data
from tqdm import tqdm
import torch
import os 
import argparse 


# TODO: Implement Style Interpolation - allow multiple styles - one content image
# TODO: Support Mask - separate foregound and background
# TODO: Support Video

def prepare_adain_model(cfg):
    # create model 
    model = get_model(cfg.MODEL.NAME, cfg)
    # load model weights
    model.load(cfg)
    # push model to device
    model.to(cfg.DEVICE)
    return model


def adain_style_transfer(model, content_img, style_img, ch, cw, alpha=1.0, logger=None, mask_img='', style_interp_weights=[]):
    with torch.no_grad():
        if mask_img:
            g_t = model.forward_with_mask(content_img, style_img, mask=mask_img, alpha=alpha)
        elif style_interp_weights:  
            g_t = model.forward_with_style_interpolation(content_img, style_img, alpha=alpha, 
            style_interp_weights=style_interp_weights)
        else:
            g_t, _, _ = model(content_img, style_img, alpha=alpha)
        _, _, new_ch, new_cw = g_t.shape 
        if ch != new_ch or cw != new_cw:
            if logger:
                logger.info('De-resize image: (%d, %d) -> (%d, %d)' % (new_ch, new_cw, ch, cw))
            else:
                print('De-resize image: (%d, %d) -> (%d, %d)' % (new_ch, new_cw, ch, cw))
            g_t = torch.nn.functional.interpolate(g_t, size=(ch, cw), mode='bilinear')
        return g_t

