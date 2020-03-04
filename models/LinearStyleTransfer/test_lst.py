from StyleTransfer.models import get_model
import torch
import numpy as np 
import time


# TODO: Support Video

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


def prepare_lst_model(cfg):
    # create model 
    model = get_model(cfg.MODEL.NAME, cfg)
    # load model weights
    model.load(cfg)
    # push model to device
    model.to(cfg.DEVICE)
    return model


def lst_style_transfer(model, content_img, style_img, ch, cw, content_seg=[], style_seg=[], logger=None, orig_content=None, test_transform=None):
    content_seg = np.asarray(content_seg)
    style_seg = np.asarray(style_seg)

    if content_seg.size > 0 and style_seg.size > 0:
        pass  # support mask
    else:
        with Timer('Elapsed time in stylization: %f', logger):
            g_t = model.forward_with_trans(content_img, style_img)  # tensor
        
        if model.cfg.MODEL.LST.DISABLE_SPN:
            pass
        else:
            orig_content = test_transform(orig_content).unsqueeze(0).to(model.cfg.DEVICE)
            with Timer('Elapsed time in propagation: %f', logger):
                g_t = model.forward_spn(g_t, orig_content)
        
        # since spn is trained, resizing g_t may cause distortion to the result image transformed by SPN
        # we choose to decode the image first, and then upsample it.
        _, _, new_ch, new_cw = g_t.shape 
        if ch != new_ch or cw != new_cw:
            if logger:
                logger.info('De-resize image: (%d, %d) -> (%d, %d)' % (new_ch, new_cw, ch, cw))
            else:
                print('De-resize image: (%d, %d) -> (%d, %d)' % (new_ch, new_cw, ch, cw))
            g_t = torch.nn.functional.interpolate(g_t, size=(ch, cw), mode='bilinear')
        return g_t


def lst_style_transfer_non_photo(model, content_img, style_img, ch, cw, alpha=1.0, logger=None, style_interp_weights=[], mask_img=None):
    if mask_img:
        return None # support mask
    elif style_interp_weights:
        return None # support weight interpolation
    else:
        with Timer('Elapsed time in stylization: %f', logger):
            g_t = model.forward_with_trans(content_img, style_img)  # tensor
        
        _, _, new_ch, new_cw = g_t.shape 
        if ch != new_ch or cw != new_cw:
            if logger:
                logger.info('De-resize image: (%d, %d) -> (%d, %d)' % (new_ch, new_cw, ch, cw))
            else:
                print('De-resize image: (%d, %d) -> (%d, %d)' % (new_ch, new_cw, ch, cw))
            g_t = torch.nn.functional.interpolate(g_t, size=(ch, cw), mode='bilinear')
        return g_t
