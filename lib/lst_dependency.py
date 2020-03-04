import torch 
import torch.nn as nn 


class LSTEncoder4(nn.Module):
    def __init__(self,):
        super(LSTEncoder4, self).__init__()
        # vgg19

        # block 1
        # 224 x 224
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1, 
            padding=0
        )
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1)) 
        # 226 x 226
        self.conv2 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        # 224 x 224

        # block 2
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        # 224 x 224

        # block 3
        self.maxPool1 = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            return_indices=False,
        )
        # 112 x 112 
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1, 
            padding=0
        )
        # 112 x 112

        # block 4
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3, 
            stride=1, 
            padding=0
        )
        # 112 x 112 

        # block 5
        self.maxPool2 = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            return_indices=False,
        )
        # 56 x 56
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(
            in_channels=128, 
            out_channels=256,
            kernel_size=3,
            stride=1, 
            padding=0
        )
        # 56 x 56

        # block 6
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        # 56 x 56 

        # block 7
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        # 56 x 56

        # block 8
        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 56 x 56 

        # block 9
        self.maxPool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=False,
        )
        # 28 x 28 
        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 28 x 28

    
    def forward(self, x, sF=None, matrix31=None):
        output = {}
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        output['r11'] = torch.nn.functional.relu(out)  # relu1-1
        
        out = self.reflecPad2(output['r11'])
        out = self.conv3(out)
        output['r12'] = torch.nn.functional.relu(out)  # relu1-2

        output['p1'] = self.maxPool1(output['r12'])
        out = self.reflecPad3(output['p1'])
        out = self.conv4(out)
        output['r21'] = torch.nn.functional.relu(out)  # relu2-1

        out = self.reflecPad4(output['r21'])
        out = self.conv5(out)
        output['r22'] = torch.nn.functional.relu(out)  # relu2-2

        output['p2'] = self.maxPool2(output['r22'])
        out = self.reflecPad5(output['p2'])
        out = self.conv6(out)
        output['r31'] = torch.nn.functional.relu(out)  # relu3-1

        if matrix31 is not None:
            feature3, transmatrix3 = matrix31(output['r31'], sF['r31'])
            out = self.reflecPad6(feature3)
        else:
            out = self.reflecPad6(output['r31'])
        out = self.conv7(out)
        output['r32'] = torch.nn.functional.relu(out)  # relu3-2

        out = self.reflecPad7(output['r32'])
        out = self.conv8(out)
        output['r33'] = torch.nn.functional.relu(out)  # relu3-3

        out = self.reflecPad8(output['r33'])
        out = self.conv9(out)
        output['r34'] = torch.nn.functional.relu(out)  # relu3-4

        output['p3'] = self.maxPool3(output['r34'])
        out = self.reflecPad9(output['p3'])
        out = self.conv10(out)
        output['r41'] = torch.nn.functional.relu(out)  # relu4-1

        return output


class LSTEncoder5(nn.Module):
    def __init__(self,):
        super(LSTEncoder5, self).__init__()
        # vgg19

        # block 1
        # 224 x 224
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1, 
            padding=0
        )
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1)) 
        # 226 x 226
        self.conv2 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        # 224 x 224

        # block 2
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        # 224 x 224

        # block 3
        self.maxPool1 = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            return_indices=False,
        )
        # 112 x 112 
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1, 
            padding=0
        )
        # 112 x 112

        # block 4
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3, 
            stride=1, 
            padding=0
        )
        # 112 x 112 

        # block 5
        self.maxPool2 = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            return_indices=False,
        )
        # 56 x 56
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(
            in_channels=128, 
            out_channels=256,
            kernel_size=3,
            stride=1, 
            padding=0
        )
        # 56 x 56

        # block 6
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        # 56 x 56 

        # block 7
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        # 56 x 56

        # block 8
        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 56 x 56 

        # block 9
        self.maxPool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=False,
        )
        # 28 x 28 
        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 28 x 28

        # block 10
        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 28 x 28 

        # block 11 
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(
            in_channels=512, 
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 28 x 28

        # block 12 
        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 28 x 28 

        # block 13
        self.maxPool4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=False,
        )
        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        # 14 x 14 
    
    def forward(self, x, sF=None, contentV256=None, styleV256=None, matrix31=None):
        output = {}
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        output['r11'] = torch.nn.functional.relu(out) # relu1-1
        
        out = self.reflecPad2(output['r11'])
        out = self.conv3(out)
        output['r12'] = torch.nn.functional.relu(out) # relu1-2

        output['p1'] = self.maxPool1(output['r12'])
        out = self.reflecPad3(output['p1'])
        out = self.conv4(out)
        output['r21'] = torch.nn.functional.relu(out) # relu2-1


        out = self.reflecPad4(output['r21'])
        out = self.conv5(out)
        output['r22'] = torch.nn.functional.relu(out) # relu2-2

        output['p2'] = self.maxPool2(output['r22'])
        out = self.reflecPad5(output['p2'])
        out = self.conv6(out)
        output['r31'] = torch.nn.functional.relu(out) # relu3-1

        if styleV256 is not None:
            feature = matrix31(
                output['r31'], sF['r31'], contentV256, styleV256)
            out = self.reflecPad6(feature)
        else:
            out = self.reflecPad6(output['r31'])

        out = self.conv7(out)
        output['r32'] = torch.nn.functional.relu(out) # relu3-2

        out = self.reflecPad7(output['r32'])
        out = self.conv8(out)
        output['r33'] = torch.nn.functional.relu(out) # relu3-3

        out = self.reflecPad8(output['r33'])
        out = self.conv9(out)
        output['r34'] = torch.nn.functional.relu(out) # relu3-4

        output['p3'] = self.maxPool3(output['r34'])
        out = self.reflecPad9(output['p3'])
        out = self.conv10(out)
        output['r41'] = torch.nn.functional.relu(out) # relu4-1

        out = self.reflecPad10(output['r41'])
        out = self.conv11(out)
        output['r42'] = torch.nn.functional.relu(out) # relu4-2

        out = self.reflecPad11(output['r42'])
        out = self.conv12(out)
        output['r43'] = torch.nn.functional.relu(out) # relu4-3

        out = self.reflecPad12(output['r43'])
        out = self.conv13(out)
        output['r44'] = torch.nn.functional.relu(out) # relu4-4

        output['p4'] = self.maxPool4(output['r44'])
        out = self.reflecPad13(output['p4'])
        out = self.conv14(out)
        output['r51'] = torch.nn.functional.relu(out) # relu5-1

        return output


class ConvNet(nn.Module):
    def __init__(self, layer, matrix_size=32):
        super(ConvNet, self).__init__()

        if layer == 'r31':
            # 256 x 64 x 64
            self.convs = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),  # (128, 64, 64)
                nn.ReLU(inplace=True), 
                nn.Conv2d(128, 64, 3, 1, 1),  # (64, 64, 64)
                nn.ReLU(inplace=True),
                nn.Conv2d(64, matrix_size, 3, 1, 1)  # (matrix_size, 64, 64)
            )
        elif layer == 'r41':
            # 512 x 32 x 32
            self.convs = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),  # (256, 32, 32)
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, 1, 1),  # (128, 32, 32)
                nn.ReLU(inplace=True),
                nn.Conv2d(128, matrix_size, 3, 1, 1)  # (matrix_size, 32, 32)
            )
        else:
            raise NotImplementedError('Expect layer to be one of r31 and r41, but got {}'.format(layer))

        self.fc = nn.Linear(matrix_size**2, matrix_size**2)
    
    def forward(self, x):
        # if x - 512 x 8 x 8
        out = self.convs(x)  # 32 x 8 x 8
        b, c, h, w = out.size()
        out = out.view(b, c, -1)  # 32 x 64
        # (32, 64) x (64, 32) - (32, 32)
        out = torch.bmm(out, out.transpose(1, 2)).div(h * w)  # 32 x 32
        out = out.view(out.size(0), -1)  # (b, 32*32)
        out = self.fc(out)  # (b, 32*32)

        return out


class TransLayer(nn.Module):
    def __init__(self, layer, matrix_size=32):
        super(TransLayer, self).__init__()
        # style net
        self.snet = ConvNet(layer, matrix_size=matrix_size)
        # content net
        self.cnet = ConvNet(layer, matrix_size=matrix_size)
        self.matrix_size = matrix_size 

        if layer == 'r31':
            self.compress = nn.Conv2d(256, matrix_size, 1, 1, 0)  # compress layer
            self.unzip = nn.Conv2d(matrix_size, 256, 1, 1, 0)  # unzip layer
        elif layer == 'r41':
            self.compress = nn.Conv2d(512, matrix_size, 1, 1, 0)  # compress layer
            self.unzip = nn.Conv2d(matrix_size, 512, 1, 1, 0)  # unzip layer 
        else:
            raise NotImplementedError('Expect layer to be one of r31 and r41, but got {}'.format(layer))

        self.trans_matrix = None
    
    def forward(self, fc, fs, trans=True):
        fc_copy = fc.clone()

        # center content feature
        cb, cc, ch, cw = fc.shape 
        fc_flat = fc.view(cb, cc, -1)
        content_mean = torch.mean(fc_flat, dim=2, keepdim=True).unsqueeze(3)  # (cb, cc, 1, 1)
        fc_centered = fc - content_mean  # (cb, cc, h1, w1)

        # center style feature
        sb, sc, sh, sw = fs.shape 
        fs_flat = fs.view(sb, sc, -1)
        style_mean = torch.mean(fs_flat, dim=2, keepdim=True).unsqueeze(3)  # (sb, sc, 1, 1)
        fs_centered = fs - style_mean   # (sb, sc, h2, w2)

        # compress content 
        content_compressed = self.compress(fc_centered)  # (cb, matrix_size, h1, w1)
        b, c, h, w = content_compressed.shape 
        content_compressed = content_compressed.view(b, c, -1)  # (cb, matrix_size, h1*w1)

        # perform transformation 
        if trans:
            cont_matrix = self.cnet(fc_centered)  # (b, matrix_size * matrix_size)
            style_matrix = self.snet(fs_centered)  # (b, matrix_size * matrix_size)

            b1, _ = cont_matrix.shape
            cont_matrix = cont_matrix.view(b1, self.matrix_size, self.matrix_size)  # (b, matrix_size, matrix_size)
            b1, _ = style_matrix.shape 
            style_matrix = style_matrix.view(b1, self.matrix_size, self.matrix_size)  # (b, matrix_size, matrix_size)

            # compute T
            trans_matrix = torch.bmm(style_matrix, cont_matrix)  # (b, matrix_size, matrix_size)
            trans_feat = torch.bmm(trans_matrix, content_compressed)   # (b, matrix_size, h1*w1)
            trans_feat = trans_feat.view(b, c, h, w)  # (b, matrix_size, h1, w1)
            
            out = self.unzip(trans_feat)  # (b, sc, h1, w1)

            out = out + style_mean  # restore mean

            return out, trans_matrix
        else:
            out = self.unzip(content_compressed.view(b, c, h, w))
            out = out + content_mean  # restore mean 
            return out



class LayerwiseLoss(nn.Module):
    def __init__(self, style_layers, content_layers, style_weight, content_weight):
        super(LayerwiseLoss, self).__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.critertion1 = nn.MSELoss(reduction='sum')
        self.critertion2 = nn.MSELoss()

    def compute_gram_matrix(self, x):
        b, c, h, w = x.shape 
        
        x_flat = x.view(b, c, -1)  # (b, c, h*w)
        # (b, c, h*w) * (b, h*w, c) -> (b, c, c)
        G = torch.bmm(x_flat, x_flat.transpose(1, 2)).div(c*h*w)
        return G 
    
    def compute_style_loss(self, input, target):
        ib, ic, ih, iw = input.shape 
        input_flat = input.view(ib, ic, -1)
        input_mean = torch.mean(input_flat, dim=2, keepdim=True)
        input_conv = self.compute_gram_matrix(input)

        tb, tc, th, tw = target.shape 
        target_flat = target.view(tb, tc, -1)
        target_mean = torch.mean(target_flat, dim=2, keepdim=True)
        target_conv = self.compute_gram_matrix(target)

        loss = self.critertion1(input_mean, target_mean) + self.critertion1(input_conv, target_conv)
        return loss / tb
    
    def forward(self, f_t, f_s, f_c):
        # content loss 
        loss_content = 0.

        for i, layer in enumerate(self.content_layers):
            f_c_layer = f_c[layer].detach()
            f_t_layer = f_t[layer]
            loss_content += self.critertion2(f_t_layer, f_c_layer)
        loss_content = loss_content * self.content_weight
        
        # style loss 
        loss_style = 0.

        for i, layer in enumerate(self.style_layers):
            f_s_layer = f_s[layer].detach()
            f_t_layer = f_t[layer]
            loss_style += self.compute_style_loss(f_t_layer, f_s_layer)

        loss_style = loss_style * self.style_weight

        loss = loss_content + loss_style

        return loss, loss_style, loss_content
