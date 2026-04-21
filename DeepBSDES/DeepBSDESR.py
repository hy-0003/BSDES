import time
import torch
import torch.nn as nn
from equations import Equation


class DeepBSDES(nn.Module):
    def __init__(self, equation:Equation, result:nn.Module, grad:nn.ModuleList, jump:nn.ModuleList, model_params:dict) -> None:
        super(DeepBSDES,self).__init__()
        self.n = model_params['n']
        self.x = model_params['x']
        self.t = model_params['t']
        self.T = model_params['T']
        self.N = model_params['N']
        self.l = model_params['l']
        self.area = model_params['area']

        self.equation = equation
        self.result = result
        self.grad = grad
        self.jump = jump
    

    def forward(self, batch_size):
        return 0, 0


class DeepBSDESR(DeepBSDES):
    def __init__(self, equation:Equation, result:nn.Module, grad:nn.ModuleList, jump:nn.ModuleList, model_params:dict) -> None:
        super(DeepBSDESR,self).__init__(equation,result,grad,jump,model_params)
    

    def forward(self, batch_size):
        delta_t = (self.T - self.t) / self.N
        theta_min, theta_max = self.area[0]
        phi_min, phi_max = self.area[1]
        theta = torch.rand(batch_size, 1) * (theta_max - theta_min) + theta_min
        phi = torch.rand(batch_size, 1) * (phi_max - phi_min) + phi_min
        self.x_polar = torch.cat([theta, phi], dim=-1)
        self.x = self.equation.tran_2to3(self.x_polar)

        process_N, process_X, discrete_t, delta_B = self.equation.get_position(
            self.n, self.x, self.t, self.T, self.N, batch_size
        )
        
        alpha = self.equation.alpha(process_N)
        beta = self.equation.beta(process_N)

        u = self.result(process_N[0], process_X[0])
        for i in range(self.N):
            grad_u = self.grad[i](process_N[i], process_X[i])
            sigma = self.equation.sigma0(process_X[i])
            sigma = sigma.transpose(-2, -1)
            T_inv = self.equation.T_inv(process_X[i])
            dB = delta_B[i].unsqueeze(-1)
            k = torch.sqrt(2*self.equation.D(process_N[i])) / self.equation.l
            k_grad_u = k * grad_u
            grad_bmm = (k_grad_u.unsqueeze(1) @ T_inv @ sigma @ dB).squeeze(-1)
            
            ui = self.jump[i](process_N[i], process_X[i])
            ui_plus_1 = self.jump[i](process_N[i] + 1, process_X[i])
            ui_minus_1 = self.jump[i](process_N[i] - 1, process_X[i])

            dis_ui_plus_1 = ui_plus_1 - ui
            dis_ui_minus_1 = ui_minus_1 - ui

            f = self.equation.f(discrete_t[i], process_N[i], process_X[i], u, grad_u)
            u = u + grad_bmm - (alpha[i] * dis_ui_plus_1 + beta[i] * dis_ui_minus_1) * delta_t - f * delta_t
            
            delta_N = (process_N[i+1] - process_N[i]).int()
            u[delta_N > 0] = u[delta_N > 0] + dis_ui_plus_1[delta_N > 0]
            u[delta_N < 0] = u[delta_N < 0] + dis_ui_minus_1[delta_N < 0]

        g = self.equation.g(process_N[self.N], process_X[self.N])
        return u, g
    


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super(MSELoss,self).__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, input, target):
        return self.loss(input,target)
    

def train(model:DeepBSDES, train_params:dict):
    epoch = train_params['epoch']
    batch_size = train_params['batch_size']
    lr = train_params['learning_rate']
    change_lr = train_params['change_lr']
    lr_change = train_params['lr_change']

    n = torch.tensor(train_params['train_valid_n']).float()
    x = torch.tensor(train_params['train_valid_x']).float()

    criterion = MSELoss()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    loss_values = torch.ones(epoch)
    result_values = torch.ones(epoch)

    # start training
    start = time.time()
    for i in range(epoch):
        if change_lr and i == int(epoch/2):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_change
        model.train()
        optimizer.zero_grad()
        u,g = model(batch_size)
        loss = criterion(u,g)
        loss.backward()
        optimizer.step()

        model.eval()
        loss_values[i] = loss.item()
        result_values[i] = model.result(n, x).item()

        print('\r%5d/{}|{}{}|{:.2f}s  [Loss: %e, Result: %7.5f]'.format(
            epoch,
            "#"*int((i+1)/epoch*50),
            " "*(50-int((i+1)/epoch*50)),
            time.time() - start) %
            (i+1,
            loss_values[i],
            result_values[i]),
            end = ' ', 
            flush=True)
    print("\nTraining has been completed.")
    return loss_values.cpu(), result_values.cpu()
