import torch

def evaluate_2d_momentum(z):
    """
    args:
        z       (bs, n, zdim)
    returns:
        system momentum
        (linear x, linear y, angular)
        all with shape (bs, n)
    """
    D = z.shape[-1] # number of state dimensions
    q = z[:, :, :D//2] # position of each mass in each direction
    qx, qy = q[:, :, ::2], q[:, :, 1::2]
    p = z[:, :, D//2:] # momentum of each mass in each direction
    px, py = p[:, :, ::2], p[:, :, 1::2]
    # evaluate cross product for angular momentum
    pang = qx * py - qy * px
    # sum across masses
    px_total, py_total = torch.sum(px, axis=-1), torch.sum(py, axis=-1)
    pang_total = torch.sum(pang, axis=-1)
    return px_total, py_total, pang_total