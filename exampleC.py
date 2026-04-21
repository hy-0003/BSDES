'''
An example of code that uses the model. 
You can define any of these components yourself.
'''
from NN_util import DMLPComplex
from DeepBSDES import DeepBSDESC as dbr
import equations as eq
import torch.nn as nn
import torch,json
import parameters.FeynmanKac.functions_params as funp
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Different training models require corresponding loading paths
path = './parameters/FeynmanKac/Backward.json'
with open(path, 'r') as f:
    params = json.load(f)
dis_params = params['Params']
equation_params = dis_params['equation_params']
model_params = dis_params['model_params']
train_params = dis_params['train_params']
dim = equation_params['dim']
N = model_params['N']
# Different training models require corresponding parameters
equation = eq.FeynmanKac(
    parameters=equation_params,
    alpha=funp.alpha_bkwd,
    beta=funp.beta_bkwd,
    g=funp.g_bkwd,
    f=funp.f_bkwd,
    U=funp.U_bkwd
)

result = DMLPComplex(
    input_dim=3,
    output_dim=1,
    hidden_dim=256,
    layer_num=5
)
# u = nn.Parameter(torch.randn(1,device=device), requires_grad=True)
grad = nn.ModuleList([
    DMLPComplex(
        input_dim=3,
        output_dim=3,
        hidden_dim=128,
        layer_num=5
    ) for _ in range(N)
])
jump = nn.ModuleList([
    DMLPComplex(
        input_dim=3,      
        output_dim=1,
        hidden_dim=128,
        layer_num=5
    ) for _ in range(N)
])
model = dbr.DeepBSDESC(
    equation=equation,
    result=result,
    grad=grad,
    jump=jump,
    model_params=model_params
)


loss_values, result_values = dbr.train(model=model, train_params=train_params)