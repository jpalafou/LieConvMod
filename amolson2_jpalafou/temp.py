SO2dynamics = Partial(SO2model, sysP=sysp[:bs])
Tdynamics = Partial(Tmodel, sysP=sysp[:bs])
z0 = z[:bs, 0, :]
zs_SO2 = odeint(SO2dynamics, z0, t[0, :T], rtol=1e-4, method='rk4')
zs_SO2 = zs_SO2.permute(1, 0, 2)
zs_T = odeint(Tdynamics, z0, t[0, :T], rtol=1e-4, method='rk4')
zs_T = zs_T.permute(1, 0, 2)
print('Ran simulation using model dynamics')