import torch
import torch.nn as nn
from torchinfo import summary

class MNISTCNN(nn.Module):
    """
    CNN architecture for MNIST digit classification
    Architecture:
    - Two convolutional blocks (conv + relu + maxpool)
    - Two fully connected layers with dropout
    - Output layer with 10 classes
    """
    
    def __init__(self, dropout_rate=0.5):
        """
        Initialize the CNN model layers
        
        Args:
            dropout_rate (float): Dropout probability for regularization
        """
        '''
        1conv => 0.9902 at 19 epochs
        2conv => 0.9937 at 10 epochs
        3conv => 0.9948 at 17 epochs
        '''
        super().__init__()
        self.conv1 = nn.Sequential(
            #input => (1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #(32, 14, 14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #(64, 7, 7)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #(128, 3, 3)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128*3*3, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10)
        )
        

    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) 
        x = x.view(x.size(0), -1)  # Flatten the features  // return batch size
        return self.fc(x)

    def summary(self, input_size=(1, 1, 28, 28)):
        """
        Print model summary including output shapes and parameters
        """
        return summary(self, input_size=input_size, col_names=["input_size", "output_size", "num_params"])