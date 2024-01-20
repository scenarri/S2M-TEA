import torch
import torch.nn as nn



class AConvNets(nn.Module):
    def __init__(self,  init_weights=False, num_classes = 10):
        super(AConvNets , self).__init__()
        self.loss_function = nn.CrossEntropyLoss()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=5, padding=0),
            #nn.BatchNorm2d(32,affine = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=6, padding=0),
            #nn.BatchNorm2d(64,affine = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=0),
            #nn.BatchNorm2d(128,affine = True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, num_classes, kernel_size=3, padding=0),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self,t):
        t = self.features(t)
        t = torch.squeeze(t,dim=2)
        t = torch.squeeze(t,dim=2)
        return t

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class AConvNets64(nn.Module):
    def __init__(self,  init_weights=False, num_classes = 10):
        super(AConvNets64 , self).__init__()
        self.loss_function = nn.CrossEntropyLoss()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            #nn.BatchNorm2d(32,affine = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            #nn.BatchNorm2d(64,affine = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            #nn.BatchNorm2d(128,affine = True),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Conv2d(128, 10, kernel_size=4, padding=0)
        if init_weights:
            self._initialize_weights()

    def forward(self,t):
        t = self.features(t)
        t = self.classifier(t)
        t = torch.squeeze(t,dim=2)
        t = torch.squeeze(t,dim=2)
        return t

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)