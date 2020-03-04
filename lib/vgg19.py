import torch.nn as nn 
import torch 


class VGG19(nn.Module):
    def __init__(self,):
        super(VGG19, self).__init__()
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
        self.relu1 = nn.ReLU()
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
        self.relu2 = nn.ReLU()
        # 224 x 224

        # block 3
        self.maxPool1 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)

        # 112 x 112 
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1, 
            padding=0
        )
        self.relu3 = nn.ReLU()
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
        self.relu4 = nn.ReLU()
        # 112 x 112 

        # block 5
        self.maxPool2 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)

        # 56 x 56
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(
            in_channels=128, 
            out_channels=256,
            kernel_size=3,
            stride=1, 
            padding=0
        )
        self.relu5 = nn.ReLU()
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
        self.relu6 = nn.ReLU()
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
        self.relu7 = nn.ReLU()
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
        self.relu8 = nn.ReLU()
        # 56 x 56 

        # block 9
        self.maxPool3 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)

        # 28 x 28 
        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.relu9 = nn.ReLU()
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
        self.relu10 = nn.ReLU()
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
        self.relu11 = nn.ReLU()
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
        self.relu12 = nn.ReLU()
        # 28 x 28 

        # block 13
        self.maxPool4 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)

        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        self.relu13 = nn.ReLU()
        # 14 x 14 
        
        # block 14
        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3, 
            stride=1,
            padding=0
        )
        self.relu14 = nn.ReLU()

        # block 15
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.relu15 = nn.ReLU()

        # block 16
        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.relu16 = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu1(out)  # relu1-1
        
        out = self.reflecPad2(out)
        out = self.conv3(out)
        out = self.relu2(out)  # relu1-2

        out = self.maxPool1(out)
        out = self.reflecPad3(out)
        out = self.conv4(out)
        out = self.relu3(out)  # relu2-1

        out = self.reflecPad4(out)
        out = self.conv5(out)
        out = self.relu4(out)  # relu2-2

        out = self.maxPool2(out)
        out = self.reflecPad5(out)
        out = self.conv6(out)
        out = self.relu5(out)  # relu3-1

        out = self.reflecPad6(out)
        out = self.conv7(out)
        out = self.relu6(out)  # relu3-2

        out = self.reflecPad7(out)
        out = self.conv8(out)
        out = self.relu7(out)  # relu3-3

        out = self.reflecPad8(out)
        out = self.conv9(out)
        out = self.relu8(out)  # relu3-4

        out = self.maxPool3(out)
        out = self.reflecPad9(out)
        out = self.conv10(out)
        out = self.relu9(out)  # relu4-1, the last layer to use in AdaIN

        out = self.reflecPad10(out)
        out = self.conv11(out)
        out = self.relu10(out)  # relu4-2

        out = self.reflecPad11(out)
        out = self.conv12(out)
        out = self.relu11(out)  # relu4-3

        out = self.reflecPad12(out)
        out = self.conv13(out)
        out = self.relu12(out)  # relu4-4

        out = self.maxPool4(out)
        out = self.reflecPad13(out)
        out = self.conv14(out)
        out = self.relu13(out)  # relu5-1

        out = self.reflecPad14(out)
        out = self.conv15(out)
        out = self.relu14(out)  # relu5-2

        out = self.reflecPad15(out)
        out = self.conv16(out)
        out = self.relu15(out)  # relu5-3

        out = self.reflecPad16(out)
        out = self.conv17(out)
        out = self.relu16(out)  # relu5-4

        return out

    def load_from_pth(self, state_dict):
        self.conv1.weight = torch.nn.Parameter(state_dict.get('0.weight').float())
        self.conv1.bias = torch.nn.Parameter(state_dict.get('0.bias').float())

        self.conv2.weight = torch.nn.Parameter(state_dict.get('2.weight').float())
        self.conv2.bias = torch.nn.Parameter(state_dict.get('2.bias').float())

        self.conv3.weight = torch.nn.Parameter(state_dict.get('5.weight').float())
        self.conv3.bias = torch.nn.Parameter(state_dict.get('5.bias').float())

        self.conv4.weight = torch.nn.Parameter(state_dict.get('9.weight').float())
        self.conv4.bias = torch.nn.Parameter(state_dict.get('9.bias').float())

        self.conv5.weight = torch.nn.Parameter(state_dict.get('12.weight').float())
        self.conv5.bias = torch.nn.Parameter(state_dict.get('12.bias').float())

        self.conv6.weight = torch.nn.Parameter(state_dict.get('16.weight').float())
        self.conv6.bias = torch.nn.Parameter(state_dict.get('16.bias').float())

        self.conv7.weight = torch.nn.Parameter(state_dict.get('19.weight').float())
        self.conv7.bias = torch.nn.Parameter(state_dict.get('19.bias').float())

        self.conv8.weight = torch.nn.Parameter(state_dict.get('22.weight').float())
        self.conv8.bias = torch.nn.Parameter(state_dict.get('22.bias').float())

        self.conv9.weight = torch.nn.Parameter(state_dict.get('25.weight').float())
        self.conv9.bias = torch.nn.Parameter(state_dict.get('25.bias').float())

        self.conv10.weight = torch.nn.Parameter(state_dict.get('29.weight').float())
        self.conv10.bias = torch.nn.Parameter(state_dict.get('29.bias').float())

        self.conv11.weight = torch.nn.Parameter(state_dict.get('32.weight').float())
        self.conv11.bias = torch.nn.Parameter(state_dict.get('32.bias').float())

        self.conv12.weight = torch.nn.Parameter(state_dict.get('35.weight').float())
        self.conv12.bias = torch.nn.Parameter(state_dict.get('35.bias').float())

        self.conv13.weight = torch.nn.Parameter(state_dict.get('38.weight').float())
        self.conv13.bias = torch.nn.Parameter(state_dict.get('38.bias').float())

        self.conv14.weight = torch.nn.Parameter(state_dict.get('42.weight').float())
        self.conv14.bias = torch.nn.Parameter(state_dict.get('42.bias').float())

        self.conv15.weight = torch.nn.Parameter(state_dict.get('45.weight').float())
        self.conv15.bias = torch.nn.Parameter(state_dict.get('45.bias').float())

        self.conv16.weight = torch.nn.Parameter(state_dict.get('48.weight').float())
        self.conv16.bias = torch.nn.Parameter(state_dict.get('48.bias').float())

        self.conv17.weight = torch.nn.Parameter(state_dict.get('51.weight').float())
        self.conv17.bias = torch.nn.Parameter(state_dict.get('51.bias').float())
