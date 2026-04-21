import torch
from equations import FokkerPlanck


def lamb_bkwd(n):
    return torch.sign(torch.relu(5-n))*torch.abs(20-n) + torch.sign(torch.relu(n-4)*torch.relu(10-n))*torch.sqrt(torch.abs(40-n)) + torch.sign(torch.relu(n-9)*torch.relu(15-n))*torch.abs(30-n) + torch.sign(torch.relu(n-14)*torch.relu(100-n))*torch.abs(100-n)/10

def alpha_bkwd(n):
    return lamb_bkwd(n)

def mu_bkwd(n):
    return torch.sign(torch.relu(10-n))*n + torch.sign(torch.relu(n-9)*torch.relu(20-n))*(n-5) + torch.sign(torch.relu(n-19)*torch.relu(30-n))*torch.sqrt(n) + torch.sign(torch.relu(n-29))*5

def beta_bkwd(n):
    return mu_bkwd(n)

def g_bkwd(n, x):
    n = n
    x = x
    x_polar = FokkerPlanck.tran_3to2(x)
    theta = x_polar[:, 0:1]
    k = torch.tensor(10.0) 
    p = torch.exp(-k * n * (theta ** 2))
    return p

def f_bkwd(t, n, x, u, grad):
    return torch.tensor(0, dtype=torch.float32)


def g_grad(n, x):
    n = n
    x = x
    x_polar = FokkerPlanck.tran_3to2(x)
    theta = x_polar[:, 0:1]
    return ((theta > 3 * torch.pi / 8) * (theta < torch.pi/2)|(theta < torch.pi / 4)).float()


def lamb_fwd(n):
    return torch.sign(torch.relu(5-n))*torch.abs(20-n) + torch.sign(torch.relu(n-4)*torch.relu(10-n))*torch.sqrt(torch.abs(40-n)) + torch.sign(torch.relu(n-9)*torch.relu(15-n))*torch.abs(30-n) + torch.sign(torch.relu(n-14)*torch.relu(100-n))*torch.abs(100-n)/10

def mu_fwd(n):
    return torch.sign(torch.relu(10-n))*n + torch.sign(torch.relu(n-9)*torch.relu(20-n))*(n-5) + torch.sign(torch.relu(n-19)*torch.relu(30-n))*torch.sqrt(n) + torch.sign(torch.relu(n-29))*5

def alpha_fwd(n):
    return mu_fwd(n+1)

def beta_fwd(n):
    res = lamb_fwd(n-1)
    res[n==0] = 0
    return res

def g_fwd(n, x):
    n = n
    x = x
    z = x[:, 2:3] 
    kappa = torch.tensor(10.0) 
    p = torch.exp(kappa * n * (z - 1.0)) 
    return p

def f_fwd(t, n, x, u, grad):
    n = n
    param = alpha_fwd(n) + beta_fwd(n) - alpha_fwd(n - 1) - beta_fwd(n + 1)
    return param * u
