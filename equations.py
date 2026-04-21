import torch


#This describes the equation of the DD model on the spherical surface.
class Equation():
    def __init__(self, g, f) -> None:
        self.g = g
        self.f = f
    
    def get_positions(self):
        return None


class FokkerPlanck(Equation):
    def __init__(self, parameters, alpha, beta, g, f):
        super(FokkerPlanck,self).__init__(g,f)
        self.D0 = parameters['D0']
        self.dim = parameters['dim']
        self.lmin = parameters['nmin']
        self.alpha = alpha
        self.beta = beta
        self.l = parameters['l']
        self.a = parameters['a']
        self.theta_min = torch.tensor(0.0)  
        self.theta_max = torch.pi*torch.tensor(1.0)
        self.phi_min = torch.tensor(0.0)
        self.phi_max = 2*torch.pi*torch.tensor(1.0)

    def D(self, n):
        return self.D0 / ((n + self.lmin)**self.a)

    def b(self, x):
        theta = x[..., 0:1]
        eps = 1e-3
        b_val = torch.cat([torch.cos(theta) / (torch.sin(theta) + eps),
                       torch.zeros_like(theta)], dim=-1)
        return b_val

    def sigma(self, x):
        theta = x[..., 0:1]
        phi = x[..., 1:2]
        row1 = torch.cat([torch.cos(theta)*torch.cos(phi),
                      torch.cos(theta)*torch.sin(phi),
                      -torch.sin(theta)], dim=-1)
        row2 = torch.cat([-torch.sin(phi)/(torch.sin(theta).abs().clip(1e-4)),
               torch.cos(phi)/(torch.sin(theta).abs().clip(1e-4)),
               torch.zeros_like(theta)], dim=-1)

        sigma_val = torch.stack([row1, row2], dim=-2)
        return sigma_val


    def next_position(self, pre_n, pre_x, delta_B, delta_t):
        alpha = self.alpha(pre_n)*delta_t
        beta = self.beta(pre_n)*delta_t
        rand = torch.rand_like(pre_n)
        delta_n = (rand < alpha).float() - (rand > 1 - beta).float()
        next_N = pre_n + delta_n
        x_polar_new = torch.sqrt(2.0*self.D(pre_n))*delta_B.clone()
        x_polar_new[:,0] += torch.pi/2
        x_dB = FokkerPlanck.tran_2to3(x_polar_new)
        x0 = FokkerPlanck.tran_2to3(torch.tensor([[torch.pi/2, 0.0]]))
        dx = x_dB - x0
        dx = (FokkerPlanck.T_inv(pre_x) @ dx.unsqueeze(-1)).squeeze(-1)
        next_X = pre_x + dx
        return next_N, next_X

    
    def get_position(self, n, x, t, T, N, size):
        delta_t = torch.tensor((T - t) / N)
        n = torch.tensor(n)
        discrete_t = torch.ones([N + 1, size, 1]) * torch.linspace(t, T, N + 1).reshape([N + 1, 1, 1])
        process_N = torch.ones([N + 1, size, 1]) * torch.randint(n[0], n[1] + 1, [size, 1])
        process_X = torch.zeros([N + 1, size, 3])
        process_X[0] = x
        delta_B = torch.randn([N, size, 2]) * torch.sqrt(delta_t)
        for i in range(N):
            process_N[i + 1], process_X[i + 1] = self.next_position(process_N[i], process_X[i], delta_B[i], delta_t)
        return process_N, process_X, discrete_t, delta_B
    

    @staticmethod
    def sigma0(x):
        x_polar = FokkerPlanck.tran_3to2(x)
        theta = torch.full_like(x_polar[..., 0:1], torch.pi/2)
        phi = torch.zeros_like(x_polar[..., 1:2]) 
        row1 = torch.cat([torch.cos(theta)*torch.cos(phi),
                        torch.cos(theta)*torch.sin(phi),
                        -torch.sin(theta)], dim=-1)
        row2 = torch.cat([-torch.sin(phi)/(torch.sin(theta).abs()),
               torch.cos(phi)/(torch.sin(theta).abs()),
               torch.zeros_like(theta)], dim=-1)

        sigma_val = torch.stack([row1, row2], dim=-2)
        return sigma_val
    
    @staticmethod
    def T_inv(x):
        x_polar = FokkerPlanck.tran_3to2(x)
        B = x.shape[0]
        ones = torch.tensor(1.0, dtype=torch.float32)
        zeros = torch.tensor(0.0, dtype=torch.float32)
        ones = ones.expand([B,1])
        zeros = zeros.expand([B,1])
        x_polar[:,0] -= torch.pi/2
        trans1 = torch.cat([torch.cat([torch.cos(x_polar[:,:1]),zeros,torch.sin(x_polar[:,:1])],dim=1).unsqueeze(1),torch.cat([zeros,ones,zeros],dim=1).unsqueeze(1),torch.cat([-torch.sin(x_polar[:,:1]),zeros,torch.cos(x_polar[:,:1])],dim=1).unsqueeze(1)],dim=1)
        trans2 = torch.cat([torch.cat([torch.cos(x_polar[:,1:]),-torch.sin(x_polar[:,1:]),zeros],dim=1).unsqueeze(1),torch.cat([torch.sin(x_polar[:,1:]),torch.cos(x_polar[:,1:]),zeros],dim=1).unsqueeze(1),torch.cat([zeros,zeros,ones],dim=1).unsqueeze(1)],dim=1)
        return torch.bmm(trans2,trans1)
    
    @staticmethod
    def tran_3to2(x_eul):
        theta = torch.acos(x_eul[:,2:3])
        phi = torch.atan2(x_eul[:,1:2], x_eul[:,0:1])
        return torch.cat([theta, phi], dim=-1)
    
    @staticmethod
    def tran_2to3(x_polar):
        theta = x_polar[..., 0:1] 
        phi = x_polar[..., 1:2]   
        x_coord = torch.sin(theta) * torch.cos(phi)
        y_coord = torch.sin(theta) * torch.sin(phi)
        z_coord = torch.cos(theta)
        x_cartesian = torch.cat([
            x_coord,
            y_coord,
            z_coord
        ], dim=-1)
        return x_cartesian
    

class FeynmanKac(FokkerPlanck):
    def __init__(self, parameters, alpha, beta, g, f, U):
        super(FeynmanKac,self).__init__(parameters,alpha,beta,g,f)
        self.U = U
    
    def get_position(self, n, x, p, t, T, N, size):
        p = torch.tensor(p)
        discrete_p = torch.ones([N+1,size,1])*torch.rand([size,1])*(p[1]-p[0])+p[0]
        process_N, process_X, discrete_t, delta_B = super().get_position(n, x, t, T, N, size)
        return process_N, process_X, discrete_p, discrete_t, delta_B
    