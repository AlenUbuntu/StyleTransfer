from StyleTransfer.models.AdaIN.adain import AdaIN
from StyleTransfer.models.WCT.wct import WCT
from StyleTransfer.models.FastPhotoStyle.fps import FastPhotoStyle
import torch

if torch.__version__ == '0.4.1':
    from StyleTransfer.models.LinearStyleTransfer.lst import LinearStyleTransfer
    model_factory = {
        'AdaIN': AdaIN,
        'WCT': WCT,
        'LST': LinearStyleTransfer,
        'FPS': FastPhotoStyle
    }
else:
    model_factory = {
        'AdaIN': AdaIN,
        'WCT': WCT,
        'FPS': FastPhotoStyle
    }


def get_model(name, *args, **kwargs):
    fac = model_factory[name]
    model = fac(*args, **kwargs)
    return model
