import torch.nn as nn
import torch




class LoRA(nn.Module):
    def __init__(self, rank, hidden_size_in, hidden_size_out, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        self.std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(rank, hidden_size_out))
        self.B = nn.Parameter(torch.zeros(hidden_size_in, rank))

    
    def forward(self, x):
        
        x = self.B @ self.A * x
        
        return x * self.alpha * 1/self.std_dev


        