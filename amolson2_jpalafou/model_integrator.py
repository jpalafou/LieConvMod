from lie_conv.dynamicsTrainer import Partial
from torchdiffeq import odeint

def model_integrator(model, sysp, bs, T):
    dynamics = Partial(model, sysP=sysp[:bs])
    z0 = z[:bs, 0, :]
    zs= odeint(dynamics, z0, t[0, :T], rtol=1e-4, method='rk4')
    zs = zs.permute(1, 0, 2)
    return zs