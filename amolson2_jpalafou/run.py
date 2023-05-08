import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Finzi imports
from lie_conv.dynamicsTrainer import Partial, HLieResNet
from lie_conv.lieGroups import Trivial, T, SO2

# amolson jpalafou imports
from model_config import num_layers, k, center
from dynamics import evaluate_2d_momentum
from model_integrator import model_integrator

# import spring dynamics data
t, z, sysp = torch.load(
    'datasets/ODEDynamics/SpringDynamics/spring_2D_10000_train.pz'
    )
t, z, sysp = t.to(torch.float32), z.to(torch.float32), sysp.to(torch.float32)
print('Imported simulation data')

# load trained model
Trivialmodel = HLieResNet(k = k, num_layers = num_layers, center = center, group=Trivial(2))
Trivialmodel.load_state_dict(
    torch.load(f"models/springmodel_Trivial.pt"
    )['model_state'])
SO2model = HLieResNet(k = k, num_layers = num_layers, center = center, group=SO2())
SO2model.load_state_dict(
    torch.load(f"models/springmodel_SO2.pt"
    )['model_state'])
Tmodel = HLieResNet(k = k, num_layers = num_layers, center = center, group=T(2))
Tmodel.load_state_dict(
    torch.load(f"models/springmodel_T.pt"
    )['model_state'])
print('Loaded models')

# run simulation using model dynamics
dataidx = 0
bs = 10
T = 200

z0 = z[:, 0, :]
zs_Trivial = model_integrator(Trivialmodel, t, z0, sysp, batch_size=bs, T=T)
zs_SO2 = model_integrator(SO2model, t, z0, sysp, batch_size=bs, T=T)
zs_T = model_integrator(Tmodel, t, z0, sysp, batch_size=bs, T=T)
print('Ran spring simulation using neural net dynamics')

# plot simulation momentums to verify that they remain constant
px, py, pang = evaluate_2d_momentum(z)
px_pred_Trivial, py_pred_Trivial, pang_pred_Trivial = evaluate_2d_momentum(zs_Trivial)
px_pred_SO2, py_pred_SO2, pang_pred_SO2 = evaluate_2d_momentum(zs_SO2)
px_pred_T, py_pred_T, pang_pred_T = evaluate_2d_momentum(zs_T)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

# x momentum
ax1.plot(t[dataidx, 0:T], px[dataidx, 0:T], label='dataset')
# ax1.plot(t[dataidx, 0:T], px_pred_Trivial[dataidx].detach().numpy(), '--', label='Trivial predicted')
ax1.plot(t[dataidx, 0:T], px_pred_SO2[dataidx].detach().numpy(), '--', label='SO2 predicted')
ax1.plot(t[dataidx, 0:T], px_pred_T[dataidx].detach().numpy(), '--', label='T2 predicted')
ax1.set_xlabel('t')
ax1.set_ylabel('linear momentum in x')
ax1.legend()

# y momentum
ax2.plot(t[dataidx, 0:T], py[dataidx, 0:T], label='dataset')
# ax2.plot(t[dataidx, 0:T], py_pred_Trivial[dataidx].detach().numpy(), '--', label='Trivial predicted')
ax2.plot(t[dataidx, 0:T], py_pred_SO2[dataidx].detach().numpy(), '--', label='SO2 predicted')
ax2.plot(t[dataidx, 0:T], py_pred_T[dataidx].detach().numpy(), '--', label='T2 predicted')
ax2.set_xlabel('t')
ax2.set_ylabel('linear momentum in y')
ax2.legend()

# angular momentum
ax3.plot(t[dataidx, 0:T], pang[dataidx, 0:T], label='dataset')
# ax3.plot(t[dataidx, 0:T], pang_pred_Trivial[dataidx].detach().numpy(), '--', label='Trivial predicted')
ax3.plot(t[dataidx, 0:T], pang_pred_SO2[dataidx].detach().numpy(), '--', label='SO2 predicted')
ax3.plot(t[dataidx, 0:T], pang_pred_T[dataidx].detach().numpy(), '--', label='T2 predicted')
ax3.set_xlabel('t')
ax3.set_ylabel('angular momentum')
ax3.legend()

plt.show()