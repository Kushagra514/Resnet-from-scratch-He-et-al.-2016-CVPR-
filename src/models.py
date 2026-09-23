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

if __name__ == "__main__":
    model = PlainCIFARNet(depth=20)

    x = torch.randn(8,3,32,32)

    output = model(x)

    print("Input:", x.shape)
    print("Output:", output.shape)
    print("Learnable layers:", model.learnable_layers)

    for depth in (20, 32, 44):
        check_model = PlainCIFARNet(depth=depth)
        check_output = check_model(x)
        assert check_output.shape == (8, 10)
        assert check_model.learnable_layers == depth



