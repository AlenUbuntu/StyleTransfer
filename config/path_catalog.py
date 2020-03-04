import os 

class DatasetCatalog(object):
    DATASETS = {
        'coco_artwiki_style_train': {
            'content': 'ms_coco/train2017',
            'style': 'wikiart/train'
        },
        'coco_artwiki_style_val': {
            'content': 'ms_coco/val2017',
            'style': 'wikiart/val'
        },
        'coco_artwiki_style_test': {
            'content': 'ms_coco/test2017',
            'style': 'wikiart/test'
        },
    }
    
    @staticmethod
    def get(name):
        if name not in DatasetCatalog.DATASETS:
            raise FileNotFoundError('{} is not found in DatasetCatalog.DATASETS.'.format(name))
        content_path = DatasetCatalog.DATASETS[name]['content']
        style_path = DatasetCatalog.DATASETS[name]['style']

        return content_path, style_path


class ModelCatalog(object):
    MODELS = {
        'vgg19-pytorch': {
            'feature_invertor_conv1_1': 'pretrain_model/vgg19/models/feature_invertor_conv1_1.pth',
            'feature_invertor_conv2_1': 'pretrain_model/vgg19/models/feature_invertor_conv2_1.pth',
            'feature_invertor_conv3_1': 'pretrain_model/vgg19/models/feature_invertor_conv3_1.pth',
            'feature_invertor_conv4_1': 'pretrain_model/vgg19/models/feature_invertor_conv4_1.pth',
            'feature_invertor_conv5_1': 'pretrain_model/vgg19/models/feature_invertor_conv5_1.pth',
            'vgg_normalised_conv1_1': 'pretrain_model/vgg19/models/vgg_normalised_conv1_1.pth',
            'vgg_normalised_conv2_1': 'pretrain_model/vgg19/models/vgg_normalised_conv2_1.pth',
            'vgg_normalised_conv3_1': 'pretrain_model/vgg19/models/vgg_normalised_conv3_1.pth',
            'vgg_normalised_conv4_1': 'pretrain_model/vgg19/models/vgg_normalised_conv4_1.pth',
            'vgg_normalised_conv5_1': 'pretrain_model/vgg19/models/vgg_normalised_conv5_1.pth',
            'vgg_normalised': 'pretrain_model/vgg19/models/vgg_normalised_1.pth',
        },
        'adain_decoder_github': 'pretrain_model/adain/decoder.pth',
        'adain_decoder_ours': 'pretrain_model/adain/adain_decoder.pth',
        'lst': {
            'vgg_r41': 'pretrain_model/lst/vgg_r41.pth',
            'vgg_r51': 'pretrain_model/lst/vgg_r51.pth',
            'vgg_r31': 'pretrain_model/lst/vgg_r31.pth',
            'lst_matrix_r31': 'pretrain_model/lst/r31.pth',
            'lst_matrix_r41': 'pretrain_model/lst/r41.pth',
            'lst_dec_r31': 'pretrain_model/lst/dec_r31.pth',
            'lst_dec_r41': 'pretrain_model/lst/dec_r41.pth',
            'lst_spn': 'pretrain_model/lst/r41_spn.pth'
        },
        'fps_photo_wct': 'pretrain_model/fps/photo_wct.pth',
    }

    @staticmethod
    def get(name1, name2=None):
        if name2:
            return ModelCatalog.MODELS[name1][name2]
        else:
            return ModelCatalog.MODELS[name1]
