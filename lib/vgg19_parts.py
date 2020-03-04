import torch.nn as nn 
import torch 


# vast majority of VGG19
class Encoder5(nn.Module):
    def __init__(self,):
        super(Encoder5, self).__init__()
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
        # 224 x 224
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
        # 226 x 226
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
        # 114 x 114
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
        # 114 x 114
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
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = torch.nn.functional.relu(out) # relu1-1
        
        out = self.reflecPad2(out)
        out = self.conv3(out)
        out = torch.nn.functional.relu(out) # relu1-2

        out = self.maxPool1(out)
        out = self.reflecPad3(out)
        out = self.conv4(out)
        out = torch.nn.functional.relu(out) # relu2-1


        out = self.reflecPad4(out)
        out = self.conv5(out)
        out = torch.nn.functional.relu(out) # relu2-2

        out = self.maxPool2(out)
        out = self.reflecPad5(out)
        out = self.conv6(out)
        out = torch.nn.functional.relu(out) # relu3-1

        out = self.reflecPad6(out)
        out = self.conv7(out)
        out = torch.nn.functional.relu(out) # relu3-2

        out = self.reflecPad7(out)
        out = self.conv8(out)
        out = torch.nn.functional.relu(out) # relu3-3

        out = self.reflecPad8(out)
        out = self.conv9(out)
        out = torch.nn.functional.relu(out) # relu3-4

        out = self.maxPool3(out)
        out = self.reflecPad9(out)
        out = self.conv10(out)
        out = torch.nn.functional.relu(out) # relu4-1

        out = self.reflecPad10(out)
        out = self.conv11(out)
        out = torch.nn.functional.relu(out) # relu4-2

        out = self.reflecPad11(out)
        out = self.conv12(out)
        out = torch.nn.functional.relu(out) # relu4-3

        out = self.reflecPad12(out)
        out = self.conv13(out)
        out = torch.nn.functional.relu(out) # relu4-4

        out = self.maxPool4(out)
        out = self.reflecPad13(out)
        out = self.conv14(out)
        out = torch.nn.functional.relu(out) # relu5-1

        return out
    
    def load_from_lua(self, vgg):
        self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())

        self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())

        self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())

        self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())

        self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
        self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())

        self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
        self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())

        self.conv7.weight = torch.nn.Parameter(vgg.get(19).weight.float())
        self.conv7.bias = torch.nn.Parameter(vgg.get(19).bias.float())

        self.conv8.weight = torch.nn.Parameter(vgg.get(22).weight.float())
        self.conv8.bias = torch.nn.Parameter(vgg.get(22).bias.float())

        self.conv9.weight = torch.nn.Parameter(vgg.get(25).weight.float())
        self.conv9.bias = torch.nn.Parameter(vgg.get(25).bias.float())

        self.conv10.weight = torch.nn.Parameter(vgg.get(29).weight.float())
        self.conv10.bias = torch.nn.Parameter(vgg.get(29).bias.float())

        self.conv11.weight = torch.nn.Parameter(vgg.get(32).weight.float())
        self.conv11.bias = torch.nn.Parameter(vgg.get(32).bias.float())

        self.conv12.weight = torch.nn.Parameter(vgg.get(35).weight.float())
        self.conv12.bias = torch.nn.Parameter(vgg.get(35).bias.float())

        self.conv13.weight = torch.nn.Parameter(vgg.get(38).weight.float())
        self.conv13.bias = torch.nn.Parameter(vgg.get(38).bias.float())

        self.conv14.weight = torch.nn.Parameter(vgg.get(42).weight.float())
        self.conv14.bias = torch.nn.Parameter(vgg.get(42).bias.float())


class Decoder5(nn.Module):
    def __init__(self,):
        super(Decoder5, self).__init__()
        # decoder 
        # 14 x 14

        # block 1
        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(
            in_channels=512, 
            out_channels=512, 
            kernel_size=3, 
            stride=1,
            padding=0
        )
        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        # block 2
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 28 x 28 

        # block 3
        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 28 x 28 

        # block 4
        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 28 x 28 

        # block 5
        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56 

        # block 6
        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 56 x 56 

        # block 7
        self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 56 x 56 

        # block 8 
        self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 56 x 56 

        # block 9
        self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112 

        # block 10 
        self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 112 x 112

        # block 11 
        self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        # block 12
        self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 224 x 224 

        # block 13
        self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 224 x 224
    
    def forward(self, x):
        out = self.reflecPad14(x)
        out = self.conv15(out)
        out = torch.nn.functional.relu(out)
        out = self.unpool1(out)  

        out = self.reflecPad15(out)
        out = self.conv16(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad16(out)
        out = self.conv17(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad17(out)
        out = self.conv18(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad18(out)
        out = self.conv19(out)
        out = torch.nn.functional.relu(out)
        out = self.unpool2(out)

        out = self.reflecPad19(out)
        out = self.conv20(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad20(out)
        out = self.conv21(out)
        out = torch.nn.functional.relu(out)
        
        out = self.reflecPad21(out)
        out = self.conv22(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad22(out)
        out = self.conv23(out)
        out = torch.nn.functional.relu(out)
        out = self.unpool3(out)

        out = self.reflecPad23(out)
        out = self.conv24(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad24(out)
        out = self.conv25(out)
        out = torch.nn.functional.relu(out)
        out = self.unpool4(out)

        out = self.reflecPad25(out)
        out = self.conv26(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad26(out)
        out = self.conv27(out)

        return out
    
    def load_from_lua(self, d):
        self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())

        self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
        self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())

        self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
        self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())

        self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
        self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())

        self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
        self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())

        self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
        self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())

        self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
        self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())

        self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
        self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())

        self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
        self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())

        self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
        self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())

        self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
        self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())

        self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
        self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())

        self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
        self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())


# vast majority of VGG19
class Encoder4(nn.Module):
    def __init__(self,):
        super(Encoder4, self).__init__()
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

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = torch.nn.functional.relu(out)  # relu1-1
        
        out = self.reflecPad2(out)
        out = self.conv3(out)
        out = torch.nn.functional.relu(out)  # relu1-2

        out = self.maxPool1(out)
        out = self.reflecPad3(out)
        out = self.conv4(out)
        out = torch.nn.functional.relu(out)  # relu2-1

        out = self.reflecPad4(out)
        out = self.conv5(out)
        out = torch.nn.functional.relu(out)  # relu2-2

        out = self.maxPool2(out)
        out = self.reflecPad5(out)
        out = self.conv6(out)
        out = torch.nn.functional.relu(out)  # relu3-1

        out = self.reflecPad6(out)
        out = self.conv7(out)
        out = torch.nn.functional.relu(out)  # relu3-2

        out = self.reflecPad7(out)
        out = self.conv8(out)
        out = torch.nn.functional.relu(out)  # relu3-3

        out = self.reflecPad8(out)
        out = self.conv9(out)
        out = torch.nn.functional.relu(out)  # relu3-4

        out = self.maxPool3(out)
        out = self.reflecPad9(out)
        out = self.conv10(out)
        out = torch.nn.functional.relu(out)  # relu4-1

        return out
    
    def load_from_lua(self, vgg):
        self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())

        self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())

        self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())

        self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())

        self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
        self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())

        self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
        self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())

        self.conv7.weight = torch.nn.Parameter(vgg.get(19).weight.float())
        self.conv7.bias = torch.nn.Parameter(vgg.get(19).bias.float())

        self.conv8.weight = torch.nn.Parameter(vgg.get(22).weight.float())
        self.conv8.bias = torch.nn.Parameter(vgg.get(22).bias.float())

        self.conv9.weight = torch.nn.Parameter(vgg.get(25).weight.float())
        self.conv9.bias = torch.nn.Parameter(vgg.get(25).bias.float())

        self.conv10.weight = torch.nn.Parameter(vgg.get(29).weight.float())
        self.conv10.bias = torch.nn.Parameter(vgg.get(29).bias.float())


class Decoder4(nn.Module):
    def __init__(self,):
        super(Decoder4, self).__init__()
        # decoder 
        # 28 x 28

        # block 1
        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56 

        # block 2
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 56 x 56 

        # block 3
        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 56 x 56 

        # block 4
        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 56 x 56 

        # block 5
        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112 

        # block 6
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 112 x 112

        # block 7
        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        # block 8
        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 224 x 224 

        # block 9
        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 224 x 224
    
    def forward(self, x):
        out = self.reflecPad10(x)
        out = self.conv11(out)
        out = torch.nn.functional.relu(out)
        out = self.unpool1(out)

        out = self.reflecPad11(out)
        out = self.conv12(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad12(out)
        out = self.conv13(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad13(out)
        out = self.conv14(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad14(out)
        out = self.conv15(out)
        out = torch.nn.functional.relu(out)
        out = self.unpool2(out)

        out = self.reflecPad15(out)
        out = self.conv16(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad16(out)
        out = self.conv17(out)
        out = torch.nn.functional.relu(out)
        out = self.unpool3(out)

        out = self.reflecPad17(out)
        out = self.conv18(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad18(out)
        out = self.conv19(out)

        return out

    def load_from_lua(self, d):
        self.conv11.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv11.bias = torch.nn.Parameter(d.get(1).bias.float())

        self.conv12.weight = torch.nn.Parameter(d.get(5).weight.float())
        self.conv12.bias = torch.nn.Parameter(d.get(5).bias.float())

        self.conv13.weight = torch.nn.Parameter(d.get(8).weight.float())
        self.conv13.bias = torch.nn.Parameter(d.get(8).bias.float())

        self.conv14.weight = torch.nn.Parameter(d.get(11).weight.float())
        self.conv14.bias = torch.nn.Parameter(d.get(11).bias.float())

        self.conv15.weight = torch.nn.Parameter(d.get(14).weight.float())
        self.conv15.bias = torch.nn.Parameter(d.get(14).bias.float())

        self.conv16.weight = torch.nn.Parameter(d.get(18).weight.float())
        self.conv16.bias = torch.nn.Parameter(d.get(18).bias.float())

        self.conv17.weight = torch.nn.Parameter(d.get(21).weight.float())
        self.conv17.bias = torch.nn.Parameter(d.get(21).bias.float())

        self.conv18.weight = torch.nn.Parameter(d.get(25).weight.float())
        self.conv18.bias = torch.nn.Parameter(d.get(25).bias.float())

        self.conv19.weight = torch.nn.Parameter(d.get(28).weight.float())
        self.conv19.bias = torch.nn.Parameter(d.get(28).bias.float())


class Encoder3(nn.Module):
    def __init__(self, ):
        super(Encoder3, self).__init__()
        # vgg
        # 224 x 224

        # block 1
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

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = torch.nn.functional.relu(out)  # relu1-1

        out = self.reflecPad2(out)
        out = self.conv3(out)
        out = torch.nn.functional.relu(out)  # relu1-2

        out = self.maxPool1(out)
        out = self.reflecPad3(out)
        out = self.conv4(out)
        out = torch.nn.functional.relu(out)  # relu2-1

        out = self.reflecPad4(out)
        out = self.conv5(out)
        out = torch.nn.functional.relu(out)  # relu2-2
        
        out = self.maxPool2(out)
        out = self.reflecPad5(out)
        out = self.conv6(out)
        out = torch.nn.functional.relu(out)  # relu3-1

        return out 
    
    def load_from_lua(self, vgg):
        self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())

        self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())

        self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())

        self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())

        self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
        self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())

        self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
        self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        # decoder
        # 56 x 56

        # block 1
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112 

        # block 2
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 112 x 112 

        # block 3 
        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        # block 4
        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 224 x 224

        # block 5
        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=0
        )
    
    def forward(self, x):
        out = self.reflecPad6(x)
        out = self.conv7(out)
        out = torch.nn.functional.relu(out)
        out = self.unpool1(out)

        out = self.reflecPad7(out)
        out = self.conv8(out)
        out = torch.nn.functional.relu(out)
        
        out = self.reflecPad8(out)
        out = self.conv9(out)
        out = torch.nn.functional.relu(out)
        out = self.unpool2(out)

        out = self.reflecPad9(out)
        out = self.conv10(out)
        out = torch.nn.functional.relu(out)
        
        out = self.reflecPad10(out)
        out = self.conv11(out)

        return out 
    
    def load_from_lua(self, d):
        self.conv7.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv7.bias = torch.nn.Parameter(d.get(1).bias.float())

        self.conv8.weight = torch.nn.Parameter(d.get(5).weight.float())
        self.conv8.bias = torch.nn.Parameter(d.get(5).bias.float())

        self.conv9.weight = torch.nn.Parameter(d.get(8).weight.float())
        self.conv9.bias = torch.nn.Parameter(d.get(8).bias.float())

        self.conv10.weight = torch.nn.Parameter(d.get(12).weight.float())
        self.conv10.bias = torch.nn.Parameter(d.get(12).bias.float())

        self.conv11.weight = torch.nn.Parameter(d.get(15).weight.float())
        self.conv11.bias = torch.nn.Parameter(d.get(15).bias.float())


class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        # vgg
        # 224 x 224

        # block 1
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
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
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = torch.nn.functional.relu(out)  # relu1-1

        out = self.reflecPad2(out)
        out = self.conv3(out)
        out = torch.nn.functional.relu(out)  # relu1-2

        out = self.maxPool1(out)
        out = self.reflecPad3(out)
        out = self.conv4(out)
        out = torch.nn.functional.relu(out)  # relu2-1

        return out 
    
    def load_from_lua(self, vgg):
        self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())

        self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())

        self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())

        self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        # decoder
        # 112 x 112

        # block 1
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224 

        # block 2
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # 224 x 224

        # block 3
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=0
        )
    
    def forward(self, x):
        out = self.reflecPad4(x)
        out = self.conv5(out)
        out = torch.nn.functional.relu(out)
        out = self.unpool1(out)

        out = self.reflecPad5(out)
        out = self.conv6(out)
        out = torch.nn.functional.relu(out)

        out = self.reflecPad6(out)
        out = self.conv7(out)

        return out 
    
    def load_from_lua(self, d):
        self.conv5.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv5.bias = torch.nn.Parameter(d.get(1).bias.float())

        self.conv6.weight = torch.nn.Parameter(d.get(5).weight.float())
        self.conv6.bias = torch.nn.Parameter(d.get(5).bias.float())

        self.conv7.weight = torch.nn.Parameter(d.get(8).weight.float())
        self.conv7.bias = torch.nn.Parameter(d.get(8).bias.float())

class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        # 224 x 224

        # block 1
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # 224 x 224
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
    
    def forward(self, x):
        out = self.conv1(x)  
        out = self.reflecPad1(out) 
        out = self.conv2(out) 
        out = torch.nn.functional.relu(out) # relu1-1

        return out 
    
    def load_from_lua(self, vgg):
        self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())

        self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
    

class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        # 224 x 224

        self.recflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=0
        )
    
    def forward(self, x):
        out = self.recflecPad2(x)
        out = self.conv3(out)

        return out 

    def load_from_lua(self, d):
        self.conv3.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv3.bias = torch.nn.Parameter(d.get(1).bias.float())
