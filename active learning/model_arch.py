import torch
import torch.nn as nn

class train1(object):

    class CustomNet(nn.Module):
        def __init__(self):
            super(train1.CustomNet, self).__init__()
            self.cnn_model = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size = 5),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(6, 16, kernel_size = 5),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2))

            self.fc_model = nn.Sequential(
                nn.Linear(7744, 1024), # (N, 2400) -> (N, 512)
                nn.ReLU(),
                nn.Linear(1024, 30))  # (N, 512)  -> (N, 30)) #30 classes

        def forward(self, x):
            # print(x.shape)
            x = self.cnn_model(x)
            # print(x.shape)    
            x = x.view(x.size(0), -1)
            # print(x.shape)    
            x = self.fc_model(x)
            # print(x.shape)
            return x
    
    def getmodel(self):
        return train1.CustomNet()

class train2(object):

    class CustomNet(nn.Module):
        def __init__(self):
            super(train2.CustomNet, self).__init__()
            self.cnn_model = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size = 5),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(6, 16, kernel_size = 5),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(16,32,kernel_size = 5),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2))

            self.fc_model = nn.Sequential(
                nn.Linear(2592, 1024), # (N, 2592) -> (N, 512)
                nn.ReLU(),
                nn.Linear(1024, 30))  # (N, 512)  -> (N, 30)) #30 classes

        def forward(self, x):
            # print(x.shape)
            x = self.cnn_model(x)
            # print(x.shape)    
            x = x.view(x.size(0), -1)
            # print(x.shape)    
            x = self.fc_model(x)
            # print(x.shape)
            return x
    
    def getmodel(self):
        return train2.CustomNet()

class train4(object):

    class CustomNet(nn.Module):
        def __init__(self):
            super(train4.CustomNet, self).__init__()
            self.cnn_model = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size = 5),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(4, 8, kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(8,16,kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(16,32,kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2))

            self.fc_model = nn.Sequential(
                nn.Linear(512, 256), # (N, 2592) -> (N, 512)
                nn.ReLU(),
                nn.Linear(256, 30))  # (N, 512)  -> (N, 30)) #30 classes

        def forward(self, x):
            x = self.cnn_model(x)
            x = x.view(x.size(0), -1)   
            x = self.fc_model(x)
            return x

    def getmodel(self):
        return train4.CustomNet()

class train6(object):

    class CustomNet(nn.Module):
        def __init__(self):
            super(train6.CustomNet, self).__init__()
            self.cnn_model = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(4, 8, kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(8,16,kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(16,32,kernel_size = 3),
                nn.ReLU())

            self.fc_model = nn.Sequential(
                nn.Linear(2048, 512), # (N, 2592) -> (N, 512)
                nn.ReLU(),
                nn.Linear(512, 30))  # (N, 512)  -> (N, 30)) #30 classes

        def forward(self, x):
            # print(x.shape)
            x = self.cnn_model(x)
            # print(x.shape) 
            x = x.view(x.size(0), -1)
            # print(x.shape)    
            x = self.fc_model(x)
            # print(x.shape)
            return x
    
    def getmodel(self):
        return train6.CustomNet()
    
class train7(object):

    class CustomNet(nn.Module):
        def __init__(self):
            super(train7.CustomNet, self).__init__()
            self.cnn_model = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.BatchNorm2d(4),
                nn.Conv2d(4, 8, kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.BatchNorm2d(8),
                nn.Conv2d(8,16,kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.BatchNorm2d(16),
                nn.Conv2d(16,32,kernel_size = 3),
                nn.ReLU())

            self.fc_model = nn.Sequential(
                nn.Linear(2048, 512), # (N, 2592) -> (N, 512)
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 30))  # (N, 512)  -> (N, 30)) #30 classes

        def forward(self, x):
            # print(x.shape)
            x = self.cnn_model(x)
            # print(x.shape) 
            x = x.view(x.size(0), -1)
            # print(x.shape)    
            x = self.fc_model(x)
            # print(x.shape)
            return x
    
    def getmodel(self):
        return train7.CustomNet()

class train8(object):
    class CustomNet(nn.Module):
        def __init__(self):
            super(train8.CustomNet, self).__init__()
            self.cnn_model = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.BatchNorm2d(4),
                nn.Conv2d(4, 8, kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.BatchNorm2d(8),
                nn.Conv2d(8,16,kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.BatchNorm2d(16),
                nn.Conv2d(16,32,kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2))

            self.fc_model = nn.Sequential(
                nn.Linear(512, 256), # (N, 2592) -> (N, 512)
                # nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(256, 30))  # (N, 512)  -> (N, 30)) #30 classes

        def forward(self, x):

            x = self.cnn_model(x)
            x = x.view(x.size(0), -1)
            x = self.fc_model(x)

            return x
    def getmodel(self):
        return train8.CustomNet()


class train9(object):
    class CustomNet(nn.Module):
        def __init__(self):
            super(train9.CustomNet, self).__init__()
            self.cnn_model = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(4, 8, kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(8,16,kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2),
                nn.Conv2d(16,32,kernel_size = 3),
                nn.ReLU(),
                nn.AvgPool2d(2, stride = 2))

            self.fc_model = nn.Sequential(
                nn.Linear(512, 256), # (N, 2592) -> (N, 512)
                nn.ReLU(),
                nn.Linear(256, 30))  # (N, 512)  -> (N, 30)) #30 classes

        def forward(self, x):
            # print(x.shape)
            x = self.cnn_model(x)
            # print(x.shape) 
            x = x.view(x.size(0), -1)
            # print(x.shape)    
            x = self.fc_model(x)
            # print(x.shape)
            return x

    def getmodel(self):
        return train9.CustomNet()


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size = 3),
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size = 3),
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,16,kernel_size = 3),
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32,kernel_size = 3),
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2))

        self.fc_model = nn.Sequential(
            nn.Linear(512, 256), # (N, 2592) -> (N, 512)
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(256, 30))  # (N, 512)  -> (N, 30)) #30 classes

    def forward(self, x):

        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)

        return x