import torch
import torch.nn as nn 
import torch.nn.functional as F 

class SummaryNet(nn.Module): 
    # Model out should be 41X68
    def __init__(self): 
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # Maxpool layer that reduces 32x32 image to 4x4
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        # self.fc = nn.Linear(in_features=6*4*4, out_features=8) 
        self.fc = nn.Linear(in_features=240, out_features=8) 

    def forward(self, x):
        x = x.view(-1, 1, 41, 68)
        x = self.pool(F.relu(self.conv1(x)))
        # x = x.view(-1, 6*4*4)
        x = x.view(-1, 240)
        x = F.relu(self.fc(x))
        return x

class Feedforward_full(nn.Module):
        def __init__(self):
            super(Feedforward_full, self).__init__()
            self.input_size = 2788
            self.hidden_size  = 256
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 8)
            self.sigmoid = torch.nn.Sigmoid()        
            
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            # output = self.sigmoid(output)
            return output
        
        
class Feedforward_foof(nn.Module):
        def __init__(self):
            super(Feedforward_foof, self).__init__()
            self.input_size = 408
            self.hidden_size  = 64
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 8)
            self.sigmoid = torch.nn.Sigmoid()        
            
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            # output = self.sigmoid(output)
            return output