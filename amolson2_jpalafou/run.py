import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Finzi imports
from lie_conv.dynamicsTrainer import Partial, HLieResNet
from lie_conv.lieGroups import Trivial, T, SO2

# amolson jpalafou imports
from model_config import num_layers, k
from dynamics import evaluate_2d_momentum
from model_integrator import model_integrator

# import spring dynamics data
t, z, sysp = torch.load(
    'datasets/ODEDynamics/SpringDynamics/spring_2D_10000_train.pz'
    )
t, z, sysp = t.to(torch.float32), z.to(torch.float32), sysp.to(torch.float32)
print('Imported simulation data')

# load trained model
SO2model = HLieResNet(k = k, num_layers = num_layers, group=SO2())
Tmodel = HLieResNet(k = k, num_layers = num_layers, group=T(2))
SO2model.load_state_dict(
    torch.load(f"models/springmodel_SO2.pt"
    )['model_state'])
Tmodel.load_state_dict(
    torch.load(f"models/springmodel_T.pt"
    )['model_state'])
print('Loaded models')

# run simulation using model dynamics
dataidx = 0
bs = 10 # evaulation batch size
T = 100 # number of simulation timesteps

# put model integrator here
zs_SO2 = model_integrator(SO2model, sysp, bs, T)
zs_T = model_integrator(Tmodel, sysp, bs, T)


# plot simulation momentums to verify that they remain constant
px, py, pang = evaluate_2d_momentum(z)
_, _, pang_pred_SO2 = evaluate_2d_momentum(zs_SO2)
_, _, pang_pred_T = evaluate_2d_momentum(zs_T)
plt.plot(t[dataidx, 0:T], pang[dataidx, 0:T], label='dataset angular momentum')
plt.plot(t[dataidx, 0:T], pang_pred_SO2[dataidx].detach().numpy(), '--', label='SO2 predicted angular momentum')
plt.plot(t[dataidx, 0:T], pang_pred_T[dataidx].detach().numpy(), '--', label='T2 predicted angular momentum')
plt.xlabel('t')
plt.title('Spring mass conservation of momentum')
plt.legend()
plt.show()