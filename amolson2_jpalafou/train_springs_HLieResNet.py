import torch
from lie_conv.dynamicsTrainer import HLieResNet
from lie_conv.lieGroups import Trivial, T, SO2
from spring_trainer import makeTrainer
from model_config import num_layers, k, group, num_epochs, n_train

group = T(2)

trainer = makeTrainer(
    num_epochs = num_epochs,
    n_train = n_train,
    network=HLieResNet,
    net_cfg = {'k':k,'num_layers':num_layers,'group':group}
    )
trainer.train(num_epochs=num_epochs)
torch.save(
    trainer.ckpt,
    f"models/springmodel_{group.__class__.__name__}.pt"
    ) # state dict