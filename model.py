import torch
import torch.nn as nn
import torch.nn.functional as F

settings = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 1000,          # Number of epochs
    "learning_rate": 0.001, # Model learning rate
    "context_length": 10,   # Context length for predictions
    "batch_size": 4,        # Number of sequences processed in parallel
}

class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32 * (settings["context_length"] - 3), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        batch, seq = x.shape
        x = x.view(batch, 1, seq) # Batch, Single univariate input feature, Seq
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1) # Flatten the tensor for ffwd
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
