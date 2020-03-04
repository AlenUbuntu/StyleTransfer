import torch 
import torch.nn as nn
import os 
from StyleTransfer.lib import VGG19
from StyleTransfer.lib import Decoder4
from StyleTransfer.config import get_encoder_model, get_decoder_model
import PIL.Image as Image
from torchvision.transforms.functional import to_tensor
torch.autograd.set_detect_anomaly(True)


class AdaIN(nn.Module):
    def __init__(self, cfg):
        super(AdaIN, self).__init__()
        self.cfg = cfg
        encoder = VGG19()


        # load encoder
        encoder_paths = get_encoder_model(cfg)
        encoder.load_from_pth(torch.load(encoder_paths[0]))

        # separate encoder by relu_x_1
        enc_layers = list(encoder.children())
        self.encoder1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.encoder2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.encoder3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.encoder4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        # fix encoder parameters
        for name in ['encoder1', 'encoder2', 'encoder3', 'encoder4']:
            for params in getattr(self, name).parameters():
                params.requires_grad = False
        
        self.decoder = Decoder4()
        self.criterion = nn.MSELoss()
        
    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, x):
        temp = x
        results = []

        for i in range(4):
            temp = getattr(self, 'encoder{}'.format(i+1))(temp)
            results.append(temp)
        
        return results
    
    # extracts relu4_1 features directly from input image
    def encode(self, x):
        temp = x 
        for i in range(4):
            temp = getattr(self, 'encoder{}'.format(i+1))(temp)
        return temp
    
    def cal_mean_std(self, x, eps=1e-5):
        # expect x.shape = (N, C, D)
        assert len(x.shape) < 4, 'AdaIN expect input tensor with shape (N, C, D), but got ({})'.format(','.join(x.shape))
        n, c = x.shape[0], x.shape[1]
        mu = torch.mean(x, dim=-1).view(n, c, 1)
        sigma = torch.sqrt(torch.var(x, dim=-1) + eps).view(n, c, 1)
        return mu, sigma  # (N, C, 1)
    
    def adaptive_instance_normalization(self, content_feat, style_feat):
        # content_feat (N, C, D1), style_feat (N, C, D2)
        # compute style mean and std 
        style_mean, style_std = self.cal_mean_std(style_feat)  # (N, C, 1)
        content_mean, content_std = self.cal_mean_std(content_feat)  # (N, C, 1)

        normalized_content_feat = (content_feat - content_mean) / content_std  # (N, C, D1)
        stylized_content = style_std * normalized_content_feat + style_mean  # (N, C, D1)
        return stylized_content

    def forward(self, content, style, alpha=1.0):
        """
        @alpha: it controls the degree of stylization. 
                alpha * stylized_content + (1 - alpha) * content is feed to decoder
        """
        assert 0 <= alpha <= 1.

        style_feature = self.encode_with_intermediate(style)  # f(s)  (N, C, H2, W2)
        content_feature = self.encode(content)  # f(c)  (N, C, H1, W1)

        # perform stylization via adaptive instance normalization 
        # we take the output of the last encoder layer
        # flatten features
        n, c = content_feature.shape[0], content_feature.shape[1]
        content_feature_flat = content_feature.view(n, c, -1)
        style_feature_flat = style_feature[-1].view(n, c, -1)
        t = self.adaptive_instance_normalization(content_feature_flat, style_feature_flat)  # (N, C, H1W1)

        t = t.view_as(content_feature)  # (N, C, H1, W1)
        # control the degree of stylization 
        # alpha * t + (1 - alpha) * f(c)
        t = alpha * t + (1 - alpha) * content_feature

        # decode via decoder 
        g_t = self.decoder(t)

        return g_t, style_feature, t  # g(t), f(s), t
    
    def forward_with_mask(self, content, style, mask, alpha=1.0):
        """
        currently only support processing content_img (1, 3, H, W) one by one
        assume mask is a PIL image
        style_img is a list of images specifying the style of each local region
        """
        content_feat = self.encode(content)  # (1, C, H1, W1)
        _, C, H, W = content_feat.shape
        content_feat_flat = content_feat.view(C, -1)  # (C, H1W1)
        # we cannot stack style_img since they may have different sizes
        style_feat = [self.encode(each_style) for each_style in style]  # [(1, C, H2, W2),..]

        # resize mask image to current content feature size
        mask = mask.resize((W, H), resample=Image.BICUBIC)  # (1, 1, H1, W1)
        mask_view = to_tensor(mask)[0].view(-1)  # note that we only take the first channel, because it is a black-white image (H1W1)
        mask_view = torch.gt(mask_view, 0.5).long()

        assert len(style_feat) == len(torch.unique(mask_view)), 'The number of input style images is expected to be {}, but got {}'.format(
            len(torch.unique(mask_view)), len(style_feat))

        # perform adain for each of the region
        target_feat = torch.zeros_like(content_feat_flat)  # (C, H1W1)
        for r in torch.unique(mask_view):
            reg_mask = mask_view == r  # (H1W1)
            reg_mask = reg_mask.expand(C, H*W)  # (C, H1W1)
            content_feat_reg = content_feat_flat[reg_mask].view(1, C, -1)  # (1, C, d1)
            style_feat_reg = style_feat[r.long()].view(1, C, -1)  # (1, C, d2)
            target_feat_reg = self.adaptive_instance_normalization(content_feat_reg, style_feat_reg)  # (1, C, d1)
            target_feat[reg_mask] = target_feat_reg.squeeze(0).view(-1)
        target_feat = target_feat.view_as(content_feat)  # (1, C, H1, W1)

        # alpha
        target_feat = alpha * target_feat + (1 - alpha) * content_feat

        # decode
        g_t = self.decoder(target_feat)
        return g_t
    
    def forward_with_style_interpolation(self, content, style, alpha=1.0, style_interp_weights=[]):
        """
        one content image, N style images, style is a list of images
        currently only support processing content image one by one
        content: (1, 3, H1, W1)
        style: [(1, 3, H2, W2), ...]
        """
        content_feat = self.encode(content)  # (1, C, H, W)
        _, C, H, W = content_feat.shape
        # note we cannot stack style images since they may have different sizes
        style_feat = [self.encode(each_style) for each_style in style]  # [(1, C, H2, W2)...]
        
        # do interpolation
        _, C, H, W = content_feat.shape 
        feat = torch.FloatTensor(1, C, H, W).zero_().to(self.cfg.DEVICE)

        for i, w in enumerate(style_interp_weights):
            # base_feat (1, C, HW)
            base_feat = self.adaptive_instance_normalization(content_feat.view(1, C, -1), style_feat[i].view(1, C, -1))
            feat = feat + w * base_feat.view(1, C, H, W)  # (1, C, H, W)
        feat = feat * alpha + content_feat * (1 - alpha)

        # decode
        g_t = self.decoder(feat)
        return g_t

    def cal_content_loss(self, f_g_t, t):
        assert f_g_t.size() == t.size()
        assert t.requires_grad == False 

        return self.criterion(f_g_t, t)  # ||f(g(t))-t||
    
    def cal_style_loss(self, g_t_feats, s_feats):
        loss = 0.

        for i in range(4):
            phi_g_t, phi_s = g_t_feats[i], s_feats[i]
            mean_phi_g_t, std_phi_g_t = self.cal_mean_std(phi_g_t)
            mean_phi_s, std_phi_s = self.cal_mean_std(phi_s)

            # ||mu(phi(g(t)))-mu(phi(s))||+||sigma(phi(g(t)))-simga(phi(s))||
            # phi - different layers of encoder f
            local_loss = self.criterion(mean_phi_g_t, mean_phi_s) + self.criterion(std_phi_g_t, std_phi_s)  
            loss += local_loss
        
        return loss 
    
    def cal_loss(self, g_t, s_feats, t):
        g_t_feats = self.encode_with_intermediate(g_t)  # f_g_t

        content_loss, style_loss = self.cal_content_loss(g_t_feats[-1], t), self.cal_style_loss(g_t_feats, s_feats)

        return content_loss, style_loss

    def load(self, cfg):
        # this is helper function to load pre-trained decoder weights provided by 
        # https://github.com/naoto0804/pytorch-AdaIN/tree/77462ebc9207b375cf246416bb929bb0d5b92241
        # load decoder
        decoder_paths = get_decoder_model(cfg)
        state_dict = torch.load(decoder_paths[0])

        # check the originality
        load_keys = [k[:k.rindex('.')] for k, v in state_dict.items()]
        load_keys = set(load_keys)

        decoder_dict = self.decoder.__dict__['_modules']
        our_keys = set([k for k, v in decoder_dict.items() if 'conv' in k])

        if our_keys == load_keys:
            # directly load state dict
            self.decoder.load_state_dict(state_dict)
        else:
            self.decoder.conv11.weight = torch.nn.Parameter(state_dict.get('1.weight').float())
            self.decoder.conv11.bias = torch.nn.Parameter(state_dict.get('1.bias').float())

            self.decoder.conv12.weight = torch.nn.Parameter(state_dict.get('5.weight').float())
            self.decoder.conv12.bias = torch.nn.Parameter(state_dict.get('5.bias').float())

            self.decoder.conv13.weight = torch.nn.Parameter(state_dict.get('8.weight').float())
            self.decoder.conv13.bias = torch.nn.Parameter(state_dict.get('8.bias').float())

            self.decoder.conv14.weight = torch.nn.Parameter(state_dict.get('11.weight').float())
            self.decoder.conv14.bias = torch.nn.Parameter(state_dict.get('11.bias').float())

            self.decoder.conv15.weight = torch.nn.Parameter(state_dict.get('14.weight').float())
            self.decoder.conv15.bias = torch.nn.Parameter(state_dict.get('14.bias').float())

            self.decoder.conv16.weight = torch.nn.Parameter(state_dict.get('18.weight').float())
            self.decoder.conv16.bias = torch.nn.Parameter(state_dict.get('18.bias').float())

            self.decoder.conv17.weight = torch.nn.Parameter(state_dict.get('21.weight').float())
            self.decoder.conv17.bias = torch.nn.Parameter(state_dict.get('21.bias').float())

            self.decoder.conv18.weight = torch.nn.Parameter(state_dict.get('25.weight').float())
            self.decoder.conv18.bias = torch.nn.Parameter(state_dict.get('25.bias').float())

            self.decoder.conv19.weight = torch.nn.Parameter(state_dict.get('28.weight').float())
            self.decoder.conv19.bias = torch.nn.Parameter(state_dict.get('28.bias').float())
            
