import torch 
import torch.nn as nn 
from StyleTransfer.lib.lst_dependency import LSTEncoder5, LSTEncoder4, TransLayer, LayerwiseLoss
from StyleTransfer.lib.vgg19_parts import Encoder3, Decoder3, Decoder4
from StyleTransfer.config import get_encoder_model, get_decoder_model, get_parts
from StyleTransfer.lib.SPN.SPN import SPN


class LinearStyleTransfer(nn.Module):
    def __init__(self, cfg):
        super(LinearStyleTransfer, self).__init__()
        self.cfg = cfg 
        disable_transfer = cfg.MODEL.LST.DISABLE_TRANSFER
        load_loss_net = cfg.MODEL.LST.LOAD_LOSS_NET
        disable_spn = cfg.MODEL.LST.DISABLE_SPN

        if cfg.MODEL.LST.LAYER == 'r31':
            self.encoder = Encoder3()
            self.decoder = Decoder3()
            if not disable_transfer:
                self.trans_layer = TransLayer(layer='r31', matrix_size=cfg.MODEL.LST.MATRIX_SIZE)
        elif cfg.MODEL.LST.LAYER == 'r41':
            self.encoder = LSTEncoder4()
            self.decoder = Decoder4()
            if not disable_transfer:
                self.trans_layer = TransLayer(layer='r41', matrix_size=cfg.MODEL.LST.MATRIX_SIZE)
        else:
            raise NotImplementedError('Invalid argument setting of MODEL.LST.LAYER: only r31 and r41 is supported.')
        
        # VGG loss module
        if load_loss_net:
            self.loss_network = LSTEncoder5()

        # load pretrained model
        encoder_paths = get_encoder_model(cfg)
        self.encoder.load_state_dict(torch.load(encoder_paths[0]))
        if load_loss_net:
            self.loss_network.load_state_dict(torch.load(encoder_paths[1]))

        decoder_paths = get_decoder_model(cfg)
        self.decoder.load_state_dict(torch.load(decoder_paths[0]))

        # disable training of encoder, decoder, loss_network
        # we only train trans_layer
        for param in self.encoder.parameters():
            param.requries_grad = False 
        
        for param in self.decoder.parameters():
            param.requires_grad = False 
        
        if load_loss_net:
            for param in self.loss_network.parameters():
                param.requires_grad = False 
        
            self.criterion = LayerwiseLoss(
                cfg.MODEL.LST.STYLE_LAYERS, 
                cfg.MODEL.LST.CONTENT_LAYERS, 
                cfg.MODEL.LST.STYLE_WEIGHT, 
                cfg.MODEL.LST.CONTENT_WEIGHT
            )

        if not disable_spn:
            self.SPN = SPN(spn=cfg.MODEL.LST.SPN_NUM)
            self.criterion_spn = nn.MSELoss(reduction='sum')

        
    def forward_with_trans(self, content, style):
        f_c = self.encoder(content)  # (N, C, H1, W1)
        f_s = self.encoder(style)  # (N, C, H1, W1)
        if self.cfg.MODEL.LST.LAYER == 'r41':
            feature, trans_matrix = self.trans_layer(f_c[self.cfg.MODEL.LST.LAYER], f_s[self.cfg.MODEL.LST.LAYER])
        else:
            feature, trans_matrix = self.trans_layer(f_c, f_s)
        t = self.decoder(feature)
    
        return t

    def cal_trans_loss(self, t, content, style):
        f_t = self.loss_network(t)
        f_s = self.loss_network(style)
        f_c = self.loss_network(content)

        loss, loss_style, loss_content = self.criterion(f_t, f_s, f_c)

        return loss, loss_style, loss_content
    
    def forward_with_no_trans(self, x):
        feat = self.encoder(x)
        x_prime = self.decoder(feat)
        return x_prime
    
    def forward_spn(self, distorted, target):
        return self.SPN(distorted, target)
    
    def cal_spn_loss(self, input, target):
        return self.criterion_spn(input, target)
    
    def load(self, cfg):
        parts_paths = get_parts(cfg)
        idx = 0
        if not cfg.MODEL.LST.DISABLE_TRANSFER:
            # load trans layer
            self.trans_layer.load_state_dict(torch.load(parts_paths[idx]))
            idx += 1
        if not cfg.MODEL.LST.DISABLE_SPN:
            # load spn 
            self.SPN.load_state_dict(torch.load(parts_paths[idx]))
