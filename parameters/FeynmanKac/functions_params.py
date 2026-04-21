import torch


def lamb_bkwd(n):
    return torch.sign(torch.relu(5-n))*torch.abs(20-n)**(1/3) + torch.sign(torch.relu(n-4)*torch.relu(10-n))*torch.sqrt(torch.abs(30-n)) + torch.sign(torch.relu(n-9)*torch.relu(15-n))*torch.abs(20-n) + torch.sign(torch.relu(n-14)*torch.relu(50-n))*torch.abs(50-n)/10

def mu_bkwd(n):
    return torch.sign(torch.relu(10-n))*1.5*n + torch.sign(torch.relu(n-9)*torch.relu(20-n))*(n-5) + torch.sign(torch.relu(n-19)*torch.relu(30-n))*torch.sqrt(torch.abs(n-10)) + torch.sign(torch.relu(n-29))*5

def alpha_bkwd(n):
    return lamb_bkwd(n)

def beta_bkwd(n):
    return mu_bkwd(n)

def U_bkwd(x):
    z = x[..., 2:3]
    U_val = 0.5 + 25.0 * (1.0 - z)
    return U_val

def g_bkwd(n, x):
    z = x[..., 2:3] 
    kappa = 0.1
    n_c = 15.0
    alpha = 1.0
    spatial_part = torch.exp(-kappa * (1.0 - z))
    state_part = 1.0 / (1.0 + torch.exp(-alpha * (n - n_c)))
    p = spatial_part * state_part
    return p + 1j * 0.0

def f_bkwd(t, n, x, p, u, grad):
    return 0



def lamb_fwd(n):
    return torch.sign(torch.relu(5-n))*torch.abs(20-n)**(1/3) + torch.sign(torch.relu(n-4)*torch.relu(10-n))*torch.sqrt(torch.abs(30-n)) + torch.sign(torch.relu(n-9)*torch.relu(15-n))*torch.abs(20-n) + torch.sign(torch.relu(n-14)*torch.relu(50-n))*torch.abs(50-n)/10

def mu_fwd(n):
    return torch.sign(torch.relu(10-n))*1.5*n + torch.sign(torch.relu(n-9)*torch.relu(20-n))*(n-5) + torch.sign(torch.relu(n-19)*torch.relu(30-n))*torch.sqrt(torch.abs(n-10)) + torch.sign(torch.relu(n-29))*5

def alpha_fwd(n):
    return mu_fwd(n+1)

def beta_fwd(n):
    res = lamb_fwd(n-1)
    res[n==0] = 0
    return res

def g_fwd(n, x):
    u_rel = torch.exp(-torch.norm(x,dim=1,keepdim=True)**2 / (n+1))
    return u_rel*(n%2) + 1j*0

def U_occupation_time(x):
    return torch.relu(torch.sign(x[:,:,:1]))

def f_fwd(t,n,x,p,u,grad):
    param = alpha_fwd(n) + beta_fwd(n) - alpha_fwd(n-1) - beta_fwd(n+1)
    return param*u