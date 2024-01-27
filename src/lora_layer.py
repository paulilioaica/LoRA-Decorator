import torch.nn as nn
import torch


class LoRALayer(nn.Module):
    def __init__(self, rank, hidden_size_in, hidden_size_out, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        self.in_features = hidden_size_in
        self.out_features = hidden_size_out

        self.std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(rank, hidden_size_out))
        self.B = nn.Parameter(torch.zeros(hidden_size_in, rank))

    
    def forward(self, x):

        if len(x.shape) == 3:
            dim1, dim2, _ = x.shape
            x = x.reshape(-1, self.in_features)
            x = x @ self.B @ self.A
            x = x.view(dim1, dim2, self.out_features)

        elif len(x.shape) <= 2:
            x = x @ self.B @ self.A

        elif len(x.shape) > 3:
            raise ValueError("LoRALayer only supports 2D or 3D inputs, got {}D".format(len(x.shape)))
        
    

        return x * self.alpha * 1 / self.std_dev

