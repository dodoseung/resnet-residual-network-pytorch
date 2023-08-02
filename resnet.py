import torch.nn.functional as F
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, input_channel=3, output_channel=10, num_blocks=18):
        super(ResNet, self).__init__()
        # Condition
        self.num_blocks = num_blocks
        assert self.num_blocks in [18, 34, 50, 101, 152], 'Please set the number of layers as 18, 34, 50, 101, 152'
        
        self.input_channel = input_channel
        self.output_channel = output_channel
        
        # Set the input layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))

        if self.num_blocks == 18:
            self.resnet = nn.ModuleList([BasicBlock(input_channel=64, output_channel=64, num_blocks=2, img_downsize=False),
                                         BasicBlock(input_channel=64, output_channel=128, num_blocks=2, img_downsize=True),
                                         BasicBlock(input_channel=128, output_channel=256, num_blocks=2, img_downsize=True),
                                         BasicBlock(input_channel=256, output_channel=512, num_blocks=2, img_downsize=True)])
            
        elif self.num_blocks == 34:
            self.resnet = nn.ModuleList([BasicBlock(input_channel=64, output_channel=64, num_blocks=3, img_downsize=False),
                                         BasicBlock(input_channel=64, output_channel=128, num_blocks=4, img_downsize=True),
                                         BasicBlock(input_channel=128, output_channel=256, num_blocks=6, img_downsize=True),
                                         BasicBlock(input_channel=256, output_channel=512, num_blocks=3, img_downsize=True)])
            
        elif self.num_blocks == 50:
            self.resnet = nn.ModuleList([BottleneckBlock(input_channel=64, in_channel=64, output_channel=256, num_blocks=3, img_downsize=False),
                                         BottleneckBlock(input_channel=256, in_channel=128, output_channel=512, num_blocks=4, img_downsize=True),
                                         BottleneckBlock(input_channel=512, in_channel=256, output_channel=1024, num_blocks=6, img_downsize=True),
                                         BottleneckBlock(input_channel=1024, in_channel=512, output_channel=2048, num_blocks=3, img_downsize=True)])
            
        # Set the output layer
        if self.num_blocks in [18, 34]:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.output_layer = nn.Linear(512, self.output_channel)
            
        elif self.num_blocks in [50, 101, 152]:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.output_layer = nn.Linear(2048, self.output_channel)


    def forward(self, x):
        # Input
        x = self.input_layer(x)

        # Resnet
        for i in range(len(self.resnet)):
            x = self.resnet[i](x)
        
        # Output
        x = self.avg_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.output_layer(x)
        x = F.softmax(x, dim=-1)
        
        return x

# Block for resnet-18 and 34
class BasicBlock(nn.Module):
    def __init__(self, input_channel=64, output_channel=64, num_blocks=2, img_downsize=False):
        super(BasicBlock, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.num_blocks = num_blocks
        self.img_downsize = img_downsize
        
        # Projection shortcut
        self.mismatch_dim = True if self.input_channel != self.output_channel else False
        self.shortcut = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
        
        # Convolution layers
        self.blocks = []
        for _ in range(self.num_blocks):
            # Downsize images with (2, 2) stride
            if self.img_downsize:
                layer = nn.Sequential(
                    nn.Conv2d(self.input_channel, self.output_channel, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                    nn.BatchNorm2d(output_channel),
                    nn.ReLU(),
                    nn.Conv2d(self.output_channel, self.output_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(output_channel))
            # Sustain the image size
            else:
                layer = nn.Sequential(
                    nn.Conv2d(self.input_channel, self.output_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(output_channel),
                    nn.ReLU(),
                    nn.Conv2d(self.output_channel, self.output_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(output_channel))
                
            self.img_downsize = False
            
            # Set the input channel same as output channel after the first layer
            self.input_channel = self.output_channel
            self.blocks.append(layer)
            
        self.blocks = nn.ModuleList(self.blocks)
        
    def forward(self, x):
        for i in range(self.num_blocks):
            if self.mismatch_dim and i == 0:
                x = self.blocks[i](x) + self.shortcut(x)
            else:
                x = self.blocks[i](x) + x
            x = F.relu(x)
            
        return x
    
# Block for resnet-50, 101, 152
class BottleneckBlock(nn.Module):
    def __init__(self, input_channel=64, in_channel=64, output_channel=64, num_blocks=2, img_downsize=False):
        super(BottleneckBlock, self).__init__()
        self.input_channel = input_channel
        self.in_channel = in_channel
        self.output_channel = output_channel
        self.num_blocks = num_blocks
        self.img_downsize = img_downsize
        
        # Projection shortcut
        self.mismatch_dim = True if self.input_channel != self.output_channel else False
        if self.input_channel == self.in_channel:
            self.shortcut = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        else:
            self.shortcut = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
        
        # Convolution layers
        self.blocks = []
        for _ in range(self.num_blocks):
            # Downsize images with (2, 2) stride
            if self.img_downsize:
                layer = nn.Sequential(
                    nn.Conv2d(self.input_channel, self.in_channel, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0)),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(),
                    nn.Conv2d(self.in_channel, self.in_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(),
                    nn.Conv2d(self.in_channel, self.output_channel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                    nn.BatchNorm2d(output_channel))
            # Sustain the image size
            else:
                layer = nn.Sequential(
                    nn.Conv2d(self.input_channel, self.in_channel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(),
                    nn.Conv2d(self.in_channel, self.in_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(),
                    nn.Conv2d(self.in_channel, self.output_channel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                    nn.BatchNorm2d(output_channel))
                
            self.img_downsize = False
            
            # Set the input channel same as output channel after the first layer
            self.input_channel = self.output_channel
            self.blocks.append(layer)
            
        self.blocks = nn.ModuleList(self.blocks)
        
    def forward(self, x):
        for i in range(self.num_blocks):
            if self.mismatch_dim and i == 0:
                x = self.blocks[i](x) + self.shortcut(x)
            else:
                x = self.blocks[i](x) + x
            x = F.relu(x)
            
        return x