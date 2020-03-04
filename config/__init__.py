from StyleTransfer.config.default import _C as cfg
from StyleTransfer.config.path_catalog import ModelCatalog, DatasetCatalog
import os


def get_data(cfg, dtype='train'):
    if dtype == 'train':
        rel_path_content, rel_path_style = DatasetCatalog.get(cfg.DATASET.TRAIN)
        return os.path.join(cfg.INPUT_DIR, rel_path_content), os.path.join(cfg.INPUT_DIR, 
        rel_path_style)
    elif dtype == 'val':
        rel_path_content, rel_path_style = DatasetCatalog.get(cfg.DATASET.VAL)
        return os.path.join(cfg.INPUT_DIR, rel_path_content), os.path.join(cfg.INPUT_DIR, 
        rel_path_style)
    else:
        rel_path_content, rel_path_style = DatasetCatalog.get(cfg.DATASET.TEST)
        return os.path.join(cfg.INPUT_DIR, rel_path_content), os.path.join(cfg.INPUT_DIR, 
        rel_path_style)


def get_encoder_model(cfg):
    paths = []
    keys = cfg.MODEL.ENCODER_NAME

    if len(keys) > 1:
        for i in range(1, len(keys)):
            paths.append(os.path.join(cfg.MODEL_DIR, ModelCatalog.get(keys[0], keys[i])))
    else:
        paths.append(os.path.join(cfg.MODEL_DIR, ModelCatalog.get(keys[0])))
    
    return paths


def get_decoder_model(cfg):
    paths = []
    keys = cfg.MODEL.DECODER_NAME

    if len(keys) > 1:
        for i in range(1, len(keys)):
            paths.append(os.path.join(cfg.MODEL_DIR, ModelCatalog.get(keys[0], keys[i])))
    else:
        paths.append(os.path.join(cfg.MODEL_DIR, ModelCatalog.get(keys[0])))
    
    return paths

def get_parts(cfg):
    paths = []
    keys = cfg.MODEL.PARTS

    if len(keys) > 1:
        for i in range(1, len(keys)):
            paths.append(os.path.join(cfg.MODEL_DIR, ModelCatalog.get(keys[0], keys[i])))
    else:
        paths.append(os.path.join(cfg.MODEL_DIR, ModelCatalog.get(keys[0])))
    
    return paths