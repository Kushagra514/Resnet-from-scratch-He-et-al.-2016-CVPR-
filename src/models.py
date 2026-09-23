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

if __name__ == "__main__":
    model = DeepPlainCNN()

    x = torch.randn(8,3,32,32)

    output = model(x)

    print("Input:", x.shape)
    print("Output:", output.shape)



