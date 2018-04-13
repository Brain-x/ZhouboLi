import numpy as np
import pymc3 as pm
import theano
import scipy.stats as stats
from multiprocessing import Process
MKL_THREADING_LAYER=GNU
exs = 4
real = 0.78
data = stats.bernoulli.rvs(p=real, size=exs)
print(data)
with pm.Model() as model1:
    theta = pm.Beta('theta', alpha=1, beta=1)
    y = pm.Bernoulli('y', p=theta, observed=data)
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(10, step=step, start=start)
    burnin = 100
    chain = trace[burnin:]
    pm.traceplot(chain, lines={'theta': real})
