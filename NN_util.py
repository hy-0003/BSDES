import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim) -> None:
        super(PositionalEncoding,self).__init__()
        self.dim = dim
        self.wk = 1 / 10**(8*torch.arange(0,dim//2,1)/dim)
        self.scale_n = nn.Parameter(torch.randn(dim) / torch.sqrt(torch.tensor(dim)),requires_grad=True)

    def forward(self, n):
        n = n
        res = torch.cat([torch.sin(n*self.wk),torch.cos(n*self.wk)],dim=1)
        return res*self.scale_n**2


class DMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 20, layer_num=3, batch_norm=False) -> None:
        super(DMLP,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.batch_norm = batch_norm
        self.PE = PositionalEncoding(
            dim=hidden_dim
        )
        self.in_layer = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU()
        )
        self.hidden_layer = nn.ModuleList(
            [(nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU()
            )) for _ in range(layer_num)]
        )
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)
        )
        if batch_norm:
            self.in_layer = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                self.in_layer
            )

    def forward(self, n, x):
        n = n
        x = x
        n = self.PE(n)
        y = self.in_layer(x)
        for i in range(self.layer_num):
            y = self.hidden_layer[i](y+n)
        return self.out_layer(y)


class DMLPComplex(DMLP):
    def __init__(self, input_dim, output_dim, hidden_dim = 20, layer_num=3, batch_norm=False):
        super(DMLPComplex,self).__init__(input_dim+1, output_dim*2, hidden_dim, layer_num, batch_norm)
        self.output_dim_rel = output_dim

    def forward(self, n, x, p):
        p_min = -35.0
        p_max = 35.0
        p_norm = 2 * (p - p_min) / (p_max - p_min) - 1
        n = n
        x = x
        p = p_norm
        output = super().forward(n,torch.cat([x,p],dim=1))
        return output[:,:self.output_dim_rel] + 1j*output[:,self.output_dim_rel:]
    

class ParameterResult(nn.Module):
    def __init__(self, dim, min=-1, max=1) -> None:
        super(ParameterResult,self).__init__()
        self.dim = dim
        self.result = nn.Parameter(data=(torch.rand(dim)*(max-min)+min),requires_grad=True)
    
    def forward(self, n, x):
        n = n
        x = x
        batch_size = n.shape[0]
        return torch.ones([batch_size,self.dim])*self.result
