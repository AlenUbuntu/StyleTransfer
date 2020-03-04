import os 
import argparse 
import torch
from torch.utils.serialization import load_lua
from StyleTransfer.lib import *



ModelDict = {
    'feature_invertor_conv1_1': Decoder1,
    'feature_invertor_conv2_1': Decoder2,
    'feature_invertor_conv3_1': Decoder3,
    'feature_invertor_conv4_1': Decoder4,
    'feature_invertor_conv5_1': Decoder5,
    'vgg_normalised_conv1_1': Encoder1,
    'vgg_normalised_conv2_1': Encoder2,
    'vgg_normalised_conv3_1': Encoder3,
    'vgg_normalised_conv4_1': Encoder4,
    'vgg_normalised_conv5_1': Encoder5,
}


def convert(path, save=True):
    src = path
    name = path.split('/')[-1].split('.')[0]
    tgt = path.replace('.t7', '.pth')

    print("Model: {}".format(name))

    model_params = load_lua(src)
    model = ModelDict[name]()
    model.load_from_lua(model_params)

    # save as pytorch models
    torch.save(model.state_dict(), tgt)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lua Model Converter')

    parser.add_argument(
        '--path',
        type=str,
        default='',
        help='path to lua model file'
    )
    args = parser.parse_args()

    convert(args.path, save=True)
