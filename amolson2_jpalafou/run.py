import torch
from lie_conv.dynamicsTrainer import FC, HLieResNet


t, z, sysp = torch.load('datasets/ODEDynamics/SpringDynamics/spring_2D_10000_train.pz')
print(t.shape)
print(z.shape)
print(sysp.shape)

model = torch.load('models/springmodel.pt')