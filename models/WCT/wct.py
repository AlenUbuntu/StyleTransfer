import torch 
import torch.nn as nn
from StyleTransfer.lib.vgg19_parts import *
from StyleTransfer.config import get_encoder_model, get_decoder_model
from torchvision.transforms.functional import to_tensor
from PIL import Image


class WCT(nn.Module):
    def __init__(self, cfg):
        super(WCT, self).__init__()
        self.cfg = cfg
        self.encoder1 = Encoder1()
        self.encoder2 = Encoder2()
        self.encoder3 = Encoder3()
        self.encoder4 = Encoder4()
        self.encoder5 = Encoder5()

        self.decoder1 = Decoder1()
        self.decoder2 = Decoder2()
        self.decoder3 = Decoder3()
        self.decoder4 = Decoder4()
        self.decoder5 = Decoder5()

        # load pre-trained weights
        encoder_paths = get_encoder_model(cfg)
        decoder_paths = get_decoder_model(cfg)

        for i, path in enumerate(encoder_paths):
            getattr(self, 'encoder{}'.format(i+1)).load_state_dict(torch.load(path))
        
        for i, path in enumerate(decoder_paths):
            getattr(self, 'decoder{}'.format(i+1)).load_state_dict(torch.load(path))
        
    
    def wct(self, cont_feat, styl_feat):
        # f_c (C, d1), f_s (C, d2)
        cFSize = cont_feat.size()
        c_mean = torch.mean(cont_feat, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean

        iden = torch.eye(cFSize[0]).to(cont_feat)  # .double()

        contentConv = torch.mm(cont_feat, cont_feat.t()
                               ).div(cFSize[1] - 1) + iden
        # del iden
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
        # c_e2, c_v = torch.eig(contentConv, True)
        # c_e = c_e2[:,0]

        k_c = cFSize[0]
        for i in range(cFSize[0] - 1, -1, -1):
            if c_e[i] >= 0.00001:
                k_c = i + 1
                break

        sFSize = styl_feat.size()
        s_mean = torch.mean(styl_feat, 1)
        styl_feat = styl_feat - s_mean.unsqueeze(1).expand_as(styl_feat)
        styleConv = torch.mm(styl_feat, styl_feat.t()).div(sFSize[1] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0] - 1, -1, -1):
            if s_e[i] >= 0.00001:
                k_s = i + 1
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cont_feat)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(
            torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + \
            s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature
    
    def transform(self, f_c, f_s, alpha):
        # convert to double 
        f_c = f_c.double()
        f_s = f_s.double()

        c, h1, w1 = f_c.shape 
        c, h2, w2 = f_s.shape 

        fc_flat = f_c.view(c, -1)  # (C, H1W1)
        fs_flat = f_s.view(c, -1)  # (C, H2W2)

        transformed_feat = self.wct(fc_flat, fs_flat)  # (C, H1W1)
        transformed_feat = transformed_feat.view_as(f_c)  # (C, H1, W1)
        
        mixed_feat = alpha * transformed_feat + (1.-alpha) * f_c

        mixed_feat = mixed_feat.float().unsqueeze(0)

        return mixed_feat
    
    def transform_with_mask(self, f_c, f_s, alpha, mask):
        # convert to double 
        # f_c - (1, C, H1, W1), f_s - [(1, C, H2, W2), ...] 
        f_c = f_c.double()
        f_s = [each.double() for each in f_s]

        _, c, h, w = f_c.shape
        f_c_flat = f_c.view(c, -1)  # (C, H1W1)

        # resize mask image to current content feature size
        mask = mask.resize((w, h), resample=Image.BICUBIC)  # (1, 1, H1, W1)
        mask_view = to_tensor(mask)[0].view(-1)  # note that we only take the first channel, because it is a black-white image (H1W1)
        mask_view = torch.gt(mask_view, 0.5).long()

        assert len(f_s) == len(torch.unique(mask_view)), 'The number of input style images is expected to be {}, but got {}'.format(
            len(torch.unique(mask_view)), len(f_s))

        # perform transformation for each of the region
        target_feat = torch.zeros_like(f_c_flat)  # (C, H1W1)
        for r in torch.unique(mask_view):
            reg_mask = mask_view == r  # (H1W1)
            reg_mask = reg_mask.expand(c, h*w)  # (C, H1W1)
            f_c_reg = f_c_flat[reg_mask].view(1, c, -1)  # (1, C, d1)
            f_s_reg = f_s[r.long()].view(1, c, -1)  # (1, C, H2W2)
            # perform style transformation
            target_feat_reg = self.wct(f_c_reg.squeeze(0), f_s_reg.squeeze(0)).unsqueeze(0)  # (1, C, d1)
            target_feat[reg_mask] = target_feat_reg.squeeze(0).view(-1)
        target_feat = target_feat.view_as(f_c) # (1, C, H1, W1)

        target_feat = alpha * target_feat + (1 - alpha) * f_c
        target_feat = target_feat.float()
        return target_feat


    def forward(self, content_img, style_img, alpha):
        # content_img - (N, C, H1, W1), style_img - (N, C, H2, W2) N = 1
        # encoder5 - decoder5
        fs_5 = self.encoder5(style_img)
        fc_5 = self.encoder5(content_img)
        fcs_5 = self.transform(fc_5.squeeze(0), fs_5.squeeze(0), alpha)

        img_5 = self.decoder5(fcs_5)

        # encoder4 - decoder4
        fs_4 = self.encoder4(style_img)
        fc_4 = self.encoder4(img_5)  # use the reconstructed content image from decoder5 
        fcs_4 = self.transform(fc_4.squeeze(0), fs_4.squeeze(0), alpha)

        img_4 = self.decoder4(fcs_4)

        # encoder3 - decoder3
        fs_3 = self.encoder3(style_img)
        fc_3 = self.encoder3(img_4)  # use the reconstructed content image from decoder4
        fcs_3 = self.transform(fc_3.squeeze(0), fs_3.squeeze(0), alpha)

        img_3 = self.decoder3(fcs_3)

        # encoder2 - decoder2 
        fs_2 = self.encoder2(style_img)
        fc_2 = self.encoder2(img_3)  # use the reconstructed content image from decoder3
        fcs_2 = self.transform(fc_2.squeeze(0), fs_2.squeeze(0), alpha)

        img_2 = self.decoder2(fcs_2)

        # encoder1 - decoder1
        fs_1 = self.encoder1(style_img)
        fc_1 = self.encoder1(img_2)  # use the reconstructed content image from decoder2
        fcs_1 = self.transform(fc_1.squeeze(0), fs_1.squeeze(0), alpha)

        img = self.decoder1(fcs_1)

        return img

    def forward_with_mask(self, content, style, mask, alpha=1.0):
        """
        currently only support processing content_img (1, 3, H, W) one by one
        assume mask is a PIL image
        style_img is a list of images specifying the style of each local region
        """
        # content_img - (N, C, H1, W1), style_img - (N, C, H2, W2) N = 1
        # encoder5 - decoder5
        fs_5 = [self.encoder5(each) for each in style]
        fc_5 = self.encoder5(content)
        fcs_5 = self.transform_with_mask(fc_5, fs_5, alpha, mask)

        img_5 = self.decoder5(fcs_5)

        # encoder4 - decoder4
        fs_4 = [self.encoder4(each) for each in style]
        fc_4 = self.encoder4(img_5)  # use the reconstructed content image from decoder5 
        fcs_4 = self.transform_with_mask(fc_4, fs_4, alpha, mask)

        img_4 = self.decoder4(fcs_4)

        # encoder3 - decoder3
        fs_3 = [self.encoder3(each) for each in style]
        fc_3 = self.encoder3(img_4)  # use the reconstructed content image from decoder4
        fcs_3 = self.transform_with_mask(fc_3, fs_3, alpha, mask)

        img_3 = self.decoder3(fcs_3)

        # encoder2 - decoder2 
        fs_2 = [self.encoder2(each) for each in style]
        fc_2 = self.encoder2(img_3)  # use the reconstructed content image from decoder3
        fcs_2 = self.transform_with_mask(fc_2, fs_2, alpha, mask)

        img_2 = self.decoder2(fcs_2)

        # encoder1 - decoder1
        fs_1 = [self.encoder1(each) for each in style]
        fc_1 = self.encoder1(img_2)  # use the reconstructed content image from decoder2
        fcs_1 = self.transform_with_mask(fc_1, fs_1, alpha, mask)

        img = self.decoder1(fcs_1)

        return img
    
    def forward_with_style_interpolation(self, content, style, alpha=1.0, style_interp_weights=[]):
        # TODO: implement style interpolation
        pass

        
