import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Finzi imports
from lie_conv.dynamicsTrainer import Partial, FC, HLieResNet
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

# load trained models
FCmodel = FC(k = k, num_layers = num_layers)
FCmodel.load_state_dict(torch.load(f"models/springmodel_FC.pt")['model_state'])
Trivialmodel = HLieResNet(
    k = k, 
    num_layers = num_layers, 
    center = center, 
    group=Trivial(2)
    )
Trivialmodel.load_state_dict(
    torch.load(f"models/springmodel_Trivial.pt"
    )['model_state'])
Tmodel = HLieResNet(k = k, num_layers = num_layers, center = center, group=T(2))
Tmodel.load_state_dict(
    torch.load(f"models/springmodel_T.pt"
    )['model_state'])
SO2model = HLieResNet(k = k, num_layers = num_layers, center = center, group=SO2())
SO2model.load_state_dict(
    torch.load(f"models/springmodel_SO2.pt"
    )['model_state'])
print('Loaded models')

# run simulation using model dynamics
dataidx = 2
bs = 10
T = 100

z0 = z[:, 0, :]
zs_FC = model_integrator(FCmodel, t, z0, sysp, batch_size=bs, T=T)
zs_Trivial = model_integrator(Trivialmodel, t, z0, sysp, batch_size=bs, T=T)
zs_T = model_integrator(Tmodel, t, z0, sysp, batch_size=bs, T=T)
zs_SO2 = model_integrator(SO2model, t, z0, sysp, batch_size=bs, T=T)
print('Ran spring simulation using neural net dynamics')

# calcualte simulation momentums
px, py, pang = evaluate_2d_momentum(z)
px_pred_FC, py_pred_FC, pang_pred_FC = evaluate_2d_momentum(zs_FC)
px_pred_Trivial, py_pred_Trivial, pang_pred_Trivial = evaluate_2d_momentum(zs_Trivial)
px_pred_T, py_pred_T, pang_pred_T = evaluate_2d_momentum(zs_T)
px_pred_SO2, py_pred_SO2, pang_pred_SO2 = evaluate_2d_momentum(zs_SO2)

# plot the dataset
plt.plot(t[dataidx, :], z[dataidx, :, 0], label=r'$m_1$')
plt.plot(t[dataidx, :], z[dataidx, :, 2], label=r'$m_2$')
plt.plot(t[dataidx, :], z[dataidx, :, 4], label=r'$m_3$')
plt.plot(t[dataidx, :], z[dataidx, :, 6], label=r'$m_4$')
plt.plot(t[dataidx, :], z[dataidx, :, 8], label=r'$m_5$')
plt.plot(t[dataidx, :], z[dataidx, :, 10], label=r'$m_6$')
plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
plt.legend()
plt.savefig('figures/dataset.png', dpi=300)
plt.close()

# plot the trajectories predicted by the models
plt.plot(z[dataidx, 0:T, 0], z[dataidx, 0:T, 12], label=r'$m_1$, dataset')
plt.plot(zs_FC[dataidx, 0:T, 0].detach().numpy(), zs_FC[dataidx, 0:T, 12].detach().numpy(), '--', label=r'$m_1$, FC')
plt.plot(zs_Trivial[dataidx, 0:T, 0].detach().numpy(), zs_Trivial[dataidx, 0:T, 12].detach().numpy(), '--', label=r'$m_1$, HLieConv-Trivial')
plt.plot(zs_T[dataidx, 0:T, 0].detach().numpy(), zs_T[dataidx, 0:T, 12].detach().numpy(), '--', label=r'$m_1$, HLieConv-T2')
plt.plot(zs_SO2[dataidx, 0:T, 0].detach().numpy(), zs_SO2[dataidx, 0:T, 12].detach().numpy(), '--', label=r'$m_1$, HLieConv-SO2')
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot{x}$')
plt.xlim([-0.1, 0.25])
plt.ylim([-1.5, 1.5])
plt.legend()
plt.savefig('figures/trajectories.png', dpi=300)
plt.close()

# plot the momentums predicted by the models
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# linear momentum
ax1.plot(t[dataidx, 0:T], px[dataidx, 0:T], label=r'dataset $p_x$')
ax1.plot(t[dataidx, 0:T], py[dataidx, 0:T], label=r'dataset $p_y$')
ax1.plot(t[dataidx, 0:T], px_pred_FC[dataidx].detach().numpy(), '--', label=r'FC $p_x$')
ax1.plot(t[dataidx, 0:T], py_pred_FC[dataidx].detach().numpy(), '--', label=r'FC $p_y$')
ax1.plot(t[dataidx, 0:T], px_pred_Trivial[dataidx].detach().numpy(), '--', label=r'HLieConv-Trivial $p_x$')
ax1.plot(t[dataidx, 0:T], py_pred_Trivial[dataidx].detach().numpy(), '--', label=r'HLieConv-Trivial $p_y$')
ax1.plot(t[dataidx, 0:T], px_pred_T[dataidx].detach().numpy(), '--', label=r'HLieConv-T2 $p_x$')
ax1.plot(t[dataidx, 0:T], py_pred_T[dataidx].detach().numpy(), '--', label=r'HLieConv-T2 $p_y$')
ax1.plot(t[dataidx, 0:T], px_pred_SO2[dataidx].detach().numpy(), '--', label=r'HLieConv-SO2 $p_x$')
ax1.plot(t[dataidx, 0:T], py_pred_SO2[dataidx].detach().numpy(), '--', label=r'HLieConv-SO2 $p_y$')
ax1.legend()

# angular momentum
ax2.plot(t[dataidx, 0:T], pang[dataidx, 0:T], label='dataset')
ax2.plot(t[dataidx, 0:T], pang_pred_T[dataidx].detach().numpy(), '--', label=r'HLieConv-T2 $p_{\theta}$')
ax2.plot(t[dataidx, 0:T], pang_pred_SO2[dataidx].detach().numpy(), '--', label=r'HLieConv-SO2 $p_{\theta}$')
ax2.set_xlabel('t')
ax2.legend()

plt.savefig('figures/momentum.png', dpi=300)
plt.close()