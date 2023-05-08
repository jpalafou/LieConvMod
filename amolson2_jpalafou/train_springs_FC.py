import argparse
import torch
from lie_conv.dynamicsTrainer import FC
from lie_conv.lieGroups import Trivial, T, SO2
from spring_trainer import makeTrainer
from model_config import num_layers, k, center, num_epochs, n_train

trainer = makeTrainer(
    num_epochs = num_epochs,
    n_train = n_train,
    network=FC,
    net_cfg = {'k':k,'num_layers':num_layers}
    )
trainer.train(num_epochs=num_epochs)
torch.save(
    trainer.ckpt,
    f"models/springmodel_FC.pt"
    ) # state dict