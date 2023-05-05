from lie_conv.dynamicsTrainer import HLieResNet
from amolson2_jpalafou.spring_trainer import makeTrainer
from amolson2_jpalafou.model_config import num_epochs, k, num_layers, n_train
from lie_conv.lieGroups import T, SO2

trainer = makeTrainer(
    num_epochs = num_epochs,
    n_train = n_train,
    network=HLieResNet,
    net_cfg = {'k':k,'num_layers':num_layers,'group':SO2()}
    )
trainer.train(num_epochs=num_epochs)
torch.save(trainer.ckpt, 'models/springmodel.pt') # state dict