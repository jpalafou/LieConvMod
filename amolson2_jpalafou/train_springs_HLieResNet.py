import argparse
import torch
from lie_conv.dynamicsTrainer import HLieResNet
from lie_conv.lieGroups import Trivial, T, SO2
from spring_trainer import makeTrainer
from model_config import num_layers, k, center, num_epochs, n_train

parser = argparse.ArgumentParser()
parser.add_argument(
    '--group',
    default=Trivial(2), 
    type=str, 
    choices = ['Trivial', 'T2', 'SO2']
    )
args = parser.parse_args()

if args.group == 'Trivial':
    group = Trivial(2)
if args.group == 'T2':
    group = T(2)
if args.group == 'SO2':
    group = SO2(0)

trainer = makeTrainer(
    num_epochs = num_epochs,
    n_train = n_train,
    network=HLieResNet,
    net_cfg = {'k':k,'num_layers':num_layers,'group':group, 'center':center}
    )
trainer.train(num_epochs=num_epochs)
torch.save(
    trainer.ckpt,
    f"models/springmodel_{group.__class__.__name__}.pt"
    ) # state dict