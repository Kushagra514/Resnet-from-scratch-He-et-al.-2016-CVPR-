import torch 
import torch.nn as nn

class PlainCNN(nn.Module):
    def __init__(self,num_classes = 10):
        super().__init__()

        self.classifier = nn.Linear(256, num_classes)
        self.features = nn.Sequential(
            # 3 -> 64 channels
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64 -> 64 
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 32 x 32 -> 16 x 16
            nn.MaxPool2d(2),

            # 64 -> 128 
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 128 -> 128
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 16 x 16 -> 8 x 8
            nn.MaxPool2d(2),

            # 128 -> 256
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 256 -> 256
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 8 x 8 -> 1 x 1
            nn.AdaptiveAvgPool2d((1,1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        return self.classifier(x)

class DeepPlainCNN(nn.Module):
    def __init__(self,num_classes = 10):
        super().__init__()

        self.features = nn.Sequential(
            #s1 1: 4 convolutions
            nn.Conv2d(3,64,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 32 x 32 -> 16 x 16
            nn.MaxPool2d(2),

            #s2 2 : 4 convs
            nn.Conv2d(64,128,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 16 x 16 -> 8 x 8
            nn.MaxPool2d(2),

            #s3 3 : 4 convs
            nn.Conv2d(128,256,kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256,256,kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256,256,kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256,256,kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1)),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        return self.classifier(x)


class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        # batch norms helps keep activations in a manageable numerical 
        # range during training and also has learnabale scale and shift 
        # parameters

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        #identity shortcut when dimensions already match
        # if we dont change the dimensions, just pass the original
        # input through unchanged

        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()

        # projection shortcut when dimensions change.
        # the dimension may not match so we need to tranform shortcut too
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self,x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # at this point, out is essentially the residual branch's
        # learned tranformation 

        out = out + identity

        out = torch.relu(out)

        return out
    
class PlainCIFARNet(nn.Module):
    def __init__(self, depth=20, num_classes=10):
        super().__init__()

        if depth < 8 or (depth - 2) % 6 != 0:
            raise ValueError("depth must satisfy depth = 6n + 2, with depth >= 8")

        blocks_per_stage = (depth - 2) // 6
        self.depth = depth
        self.blocks_per_stage = blocks_per_stage

        self.initial = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.stage1 = self._make_stage(16, 16, blocks_per_stage)
        self.stage2 = self._make_stage(16, 32, blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(32, 64, blocks_per_stage, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def _make_stage(self, in_channels, out_channels, blocks, stride=1):
        layers = [PlainBlock(in_channels, out_channels, stride)]
        layers.extend(
            PlainBlock(out_channels, out_channels)
            for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    @property
    def convolutional_layers(self):
        return sum(isinstance(module, nn.Conv2d) for module in self.modules())

    @property
    def learnable_layers(self):
        return self.convolutional_layers + 1
    
    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class ResNetCIFAR(nn.Module):
    def __init__(self,depth=20,num_classes=10):
        super().__init__()

        if depth < 8 or (depth - 2) % 6 != 0:
            raise ValueError(
                "depth must satisy depth = 6n + 2"
            )

        blocks_per_stage = (depth-2) // 6

        self.depth =depth
        self.blocks_per_stage = blocks_per_stage

        #initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        #three stages of residual blocks
        self.stage1 = self._make_stage(
            16,16,blocks_per_stage,stride=1
        )

        self.stage2 = self._make_stage(
            16,32,blocks_per_stage,stride=2
        )

        self.stage3 = self._make_stage(
            32,64,blocks_per_stage,stride=2
        )

        #global average pooling 
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        #CIFAR-10 classifier
        self.classifier = nn.Linear(64,num_classes)


    def _make_stage(
            self,
            in_channels,
            out_channels,
            blocks,
            stride=1,
    ):
        layers = [
            ResidualBlock(
                in_channels,
                out_channels,
                stride = stride
            )
        ]

        for _ in range(blocks - 1):
            layers.append(
                ResidualBlock(
                    out_channels,
                    out_channels
                )
            )
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.initial(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.pool(x)
        x = torch.flatten(x,1)

        return self.classifier(x)

if __name__ == "__main__":

    x = torch.randn(8, 3, 32, 32)

    # Test PlainCIFARNet
    for depth in (20, 32, 44):
        check_model = PlainCIFARNet(depth=depth)
        check_output = check_model(x)

        assert check_output.shape == (8, 10)

        print(
            f"PlainCIFARNet-{depth}: "
            f"output={check_output.shape}"
        )

    # Test ResNetCIFAR
    for depth in (20, 32, 44):
        check_model = ResNetCIFAR(depth=depth)
        check_output = check_model(x)

        assert check_output.shape == (8, 10)

        print(
            f"ResNetCIFAR-{depth}: "
            f"output={check_output.shape}"
        )



