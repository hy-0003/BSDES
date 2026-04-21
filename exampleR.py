'''
An example of code that uses the model. 
You can define any of these components yourself.
'''
from NN_util import DMLP, ParameterResult
from DeepBSDES import DeepBSDESR as dbr
import equations as eq
import torch.nn as nn
import torch,json
import parameters.FokkerPlanck.functions_params as funp
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Different training models require corresponding loading paths
path = './parameters/FokkerPlanck/Backward.json'
with open(path, 'r') as f:
    params = json.load(f)
dis_params = params['Params']
equation_params = dis_params['equation_params']
model_params = dis_params['model_params']
train_params = dis_params['train_params']
dim = equation_params['dim']
N = model_params['N']

# Different training models require corresponding parameters
equation = eq.FokkerPlanck(
    parameters=equation_params,
    alpha=funp.alpha_bkwd,
    beta=funp.beta_bkwd,
    g=funp.g_bkwd,
    f=funp.f_bkwd
)
result = DMLP(
    input_dim=3, 
    output_dim=1,
    hidden_dim=32,
    layer_num=3
)
# result = ParameterResult(dim=1,min=0.65,max=0.75)
grad = nn.ModuleList([
    DMLP(
        input_dim=3,
        output_dim=3,
        hidden_dim=32,
        layer_num=3
    ) for _ in range(N)
])
jump = nn.ModuleList([
    DMLP(
        input_dim=3,      
        output_dim=1,
        hidden_dim=32,
        layer_num=3
    ) for _ in range(N)
])
model = dbr.DeepBSDESR(
    equation=equation,
    result=result,
    grad=grad,
    jump=jump,
    model_params=model_params
)


loss_values, result_values = dbr.train(model=model, train_params=train_params)