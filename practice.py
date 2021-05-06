import scipy.stats as ss
import numpy as np

#%% p.117
ss.norm.cdf(0.8, loc=0, scale=1)
ss.norm.cdf(0.6, loc=0, scale=1) - ss.norm.cdf(-0.3, loc=0, scale=1)
ss.norm.ppf(0.7, loc=0, scale=1)

ss.norm.sf(0, loc=3, scale=4)
ss.norm.cdf(5, loc=3, scale=4) - ss.norm.cdf(1, loc=3, scale=4)
ss.norm(loc=3, scale=4).isf(0.7)

#%% p.225
ss.f(22, 17).ppf(0.05)

#%% p.231, 235
data = np.array([52, 53, 44, 46, 48, 55, 51, 48, 46, 47])
n = len(data)
x_mean = data.mean()
x_min = x_mean - 1.96*np.sqrt(1.2)
x_max = x_mean + 1.96*np.sqrt(1.2)
print(x_min, '< μ <', x_max)

t_min, t_max = ss.t(n - 1).interval(0.95)
x_var = data.var(ddof=1)
x_min = x_mean + t_min*np.sqrt(x_var/n)
x_max = x_mean + t_max*np.sqrt(x_var/n)
print(x_min, '< μ <', x_max)
