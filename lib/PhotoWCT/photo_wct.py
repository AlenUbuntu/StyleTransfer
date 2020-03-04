"""
The implementation of the function is heavily borrowed from 
https://github.com/NVIDIA/FastPhotoStyle/blob/af0c8fecce58aa71f76488546231214f6684be02/photo_wct.py#L13
We thank NVIDIA for sharing this code.
"""

import numpy as np 
from PIL import Image 
import torch 
import torch.nn as nn 
from .models import VGGEncoder, VGGDecoder


class PhotoWCT(nn.Module):
    def __init__(self, cfg):
        super(PhotoWCT, self).__init__()
        self.e1 = VGGEncoder(level=1)
        self.d1 = VGGDecoder(level=1)

        self.e2 = VGGEncoder(level=2)
        self.d2 = VGGDecoder(level=2)

        self.e3 = VGGEncoder(level=3)
        self.d3 = VGGDecoder(level=3)

        self.e4 = VGGEncoder(level=4)
        self.d4 = VGGDecoder(level=4)

        self.label_set = None
        self.label_indicator = None
        self.cfg = cfg
    
    def transform(self, cont_img, styl_img, cont_seg, styl_seg):
        self.__compute_label_info(cont_seg, styl_seg)
        sF4, sF3, sF2, sF1 = self.e4.forward_multiple(styl_img)

        cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = self.e4(
            cont_img)
        sF4 = sF4.data.squeeze(0)
        cF4 = cF4.data.squeeze(0)
        csF4 = self.__feature_wct(cF4, sF4, cont_seg, styl_seg)
        Im4 = self.d4(csF4, cpool_idx, cpool1, cpool_idx2,
                      cpool2, cpool_idx3, cpool3)

        cF3, cpool_idx, cpool1, cpool_idx2, cpool2 = self.e3(Im4)
        sF3 = sF3.data.squeeze(0)
        cF3 = cF3.data.squeeze(0)
        csF3 = self.__feature_wct(cF3, sF3, cont_seg, styl_seg)
        Im3 = self.d3(csF3, cpool_idx, cpool1, cpool_idx2, cpool2)

        cF2, cpool_idx, cpool = self.e2(Im3)
        sF2 = sF2.data.squeeze(0)
        cF2 = cF2.data.squeeze(0)
        csF2 = self.__feature_wct(cF2, sF2, cont_seg, styl_seg)
        Im2 = self.d2(csF2, cpool_idx, cpool)

        cF1 = self.e1(Im2)
        sF1 = sF1.data.squeeze(0)
        cF1 = cF1.data.squeeze(0)
        csF1 = self.__feature_wct(cF1, sF1, cont_seg, styl_seg)
        Im1 = self.d1(csF1)
        return Im1


    def __wct_core(self, cont_feat, styl_feat):
        cFSize = cont_feat.size()
        c_mean = torch.mean(cont_feat, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean

        iden = torch.eye(cFSize[0])  # .double()
        if self.is_cuda:
            iden = iden.cuda()

        contentConv = torch.mm(cont_feat, cont_feat.t()
                               ).div(cFSize[1] - 1) + iden
        # del iden
        # contentConv = torch.mm(cont_feat, cont_feat.t()
        #                        ).div(cFSize[1] - 1)
        c_u, c_e, c_v = torch.svd(contentConv, some=False)

        # much slower
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

        # much slower
        # s_e2, s_v = torch.eig(styleConv, True)
        # s_e = s_e2[:, 0]

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

    def __compute_label_info(self, cont_seg, styl_seg):
        if cont_seg.size == False or styl_seg.size == False:
            return
        max_label = np.max(cont_seg) + 1
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(max_label)
        for l in self.label_set:
            # the corresponding region in content seg image and style seg image cannot be too
            # far away from each other. 
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            
            o_cont_mask = np.where(cont_seg.reshape(
                cont_seg.shape[0] * cont_seg.shape[1]) == l)
            
            o_styl_mask = np.where(styl_seg.reshape(
                styl_seg.shape[0] * styl_seg.shape[1]) == l)
            
            self.label_indicator[l] = is_valid(
                o_cont_mask[0].size, o_styl_mask[0].size)
            

    def __feature_wct(self, cont_feat, styl_feat, cont_seg, styl_seg):
        cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2)
        styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2)

        cont_feat_flat = cont_feat.view(cont_c, -1).clone().to(cont_feat)
        styl_feat_flat = styl_feat.view(styl_c, -1).clone().to(styl_feat)

        if cont_seg.size == False or styl_seg.size == False:
            target_feature = self.__wct_core(cont_feat_flat, styl_feat_flat)
        else:
            target_feature = cont_feat.view(cont_c, -1).clone()

            if len(cont_seg.shape) == 2:  # gray-scale
                t_cont_seg = np.asarray(Image.fromarray(
                    cont_seg).resize((cont_w, cont_h), Image.NEAREST))
            else:  # rgb
                t_cont_seg = np.asarray(Image.fromarray(
                    cont_seg, mode='RGB').resize((cont_w, cont_h), Image.NEAREST))
            
            if len(styl_seg.shape) == 2:  # gray-scale
                t_styl_seg = np.asarray(Image.fromarray(
                    styl_seg).resize((styl_w, styl_h), Image.NEAREST))
            else:  # rgb
                t_styl_seg = np.asarray(Image.fromarray(
                    styl_seg, mode='RGB').resize((styl_w, styl_h), Image.NEAREST))
            
            for l in self.label_set:
                # if the size of the semantic region in the content and style images 
                # significantly vary from each other, we simply ignore it
                if self.label_indicator[l] == 0:
                    continue

                cont_mask = np.where(t_cont_seg.reshape(
                    t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l)  # (index, element)
                styl_mask = np.where(t_styl_seg.reshape(
                    t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l)  # (index, element)
                
                if cont_mask[0].size <= 0 or styl_mask[0].size <= 0:
                    continue
                
                cont_indi = torch.LongTensor(cont_mask[0]).to(self.cfg.DEVICE)
                styl_indi = torch.LongTensor(styl_mask[0]).to(self.cfg.DEVICE)

                cont_feat_FG = torch.index_select(cont_feat_flat, 1, cont_indi)
                styl_feat_FG = torch.index_select(styl_feat_flat, 1, styl_indi)

                tmp_target_feat = self.__wct_core(cont_feat_FG, styl_feat_FG)

                if torch.__version__ >= "0.4.0":
                    new_target_feature = torch.transpose(target_feature, 1, 0)
                    new_target_feature.index_copy_(0, cont_indi,
                                                   torch.transpose(tmp_target_feat, 1, 0))
                    target_feature = torch.transpose(new_target_feature, 1, 0)
                else:
                    target_feature.index_copy_(
                        1, cont_indi, tmp_target_feat)
        
        target_feature = target_feature.view_as(cont_feat)  # (C, H, W)
        ccsF = target_feature.float().unsqueeze(0)  # (1, C, H, W)
        return ccsF

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, *input):
        pass
