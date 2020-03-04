import torch.nn as nn 
import torch
from StyleTransfer.lib.PhotoWCT.photo_wct import PhotoWCT
from StyleTransfer.lib.PhotoWCT.photo_gif import GIFSmoothing
from StyleTransfer.lib.PhotoWCT.photo_smooth import Propagator
from StyleTransfer.config import get_parts


class FastPhotoStyle(nn.Module):
    def __init__(self, cfg):
        super(FastPhotoStyle, self).__init__()
        self.cfg = cfg 

        self.photo_wct = PhotoWCT(cfg)
        
        if self.cfg.MODEL.FPS.FAST:
            self.photo_smooth = GIFSmoothing(self.cfg.MODEL.FPS.R, self.cfg.MODEL.FPS.EPS)
        else:
            self.photo_smooth = Propagator(self.cfg.MODEL.FPS.BETA)
        
    def transform(self, cont_img, styl_img, cont_seg, styl_seg):
        return self.photo_wct.transform(cont_img, styl_img, cont_seg, styl_seg)
    
    def smooth(self, input, target):
        return self.photo_smooth.process(input, target)
    
    def load(self, cfg):
        parts_paths = get_parts(cfg)
        self.photo_wct.load_state_dict(torch.load(parts_paths[0]))
