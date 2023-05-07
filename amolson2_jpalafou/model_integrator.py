from lie_conv.dynamicsTrainer import Partial
from torchdiffeq import odeint

def model_integrator(model, t, z0, sysp, batch_size, T):
    dynamics = Partial(model, sysP=sysp[:batch_size])
    zs= odeint(dynamics, z0[:batch_size], t[0, :T], rtol=1e-4, method='rk4')
    zs = zs.permute(1, 0, 2)
    return zs