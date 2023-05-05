import numpy as np
import torch
import matplotlib.pyplot as plt
from lie_conv.dynamicsTrainer import HLieResNet
from amolson2_jpalafou.model_config import k, num_layers

# import spring dynamcis data
t, z, sysp = torch.load('datasets/ODEDynamics/SpringDynamics/spring_2D_10000_train.pz')
t, z, sysp = t.to(torch.float32), z.to(torch.float32), sysp.to(torch.float32)

# load trained model
model = HLieResNet(k = k, num_layers = num_layers)
model.load_state_dict(torch.load('models/springmodel.pt')['model_state'])

# evaluate model at all subsequent timesteps
batch_idx = 1
model_evals = []
for k in range(499):
    output = model(torch.Tensor([]), z[0:5, k, :], sysp[0:5]).detach().numpy()
    model_evals.append(output)
model_evals = np.asarray(model_evals)

# plot dynamics data against model prediction
midx = 2 # up to 5
plt.plot(t[batch_idx], z[batch_idx, :, midx + 12], label='x dot, m1')
plt.plot(t[batch_idx, 1:], model_evals[:, batch_idx, midx + 0], '--', label='model x dot, m1')
plt.plot(t[batch_idx], z[batch_idx, :, midx + 18], label='y dot, m1')
plt.plot(t[batch_idx, 1:], model_evals[:, batch_idx, midx + 6], '--', label='model y dot, m1')
plt.grid()
plt.legend()
plt.show()