import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Hiragino Sans'
from sklearn import linear_model as lm
import statsmodels.formula.api as smf

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


#%% p8
data = [[1, 8], [3, 6], [3, 5], [4, 7], [6, 6], [7, 5], [7, 6], [9, 5]]
df = pd.DataFrame(data, columns=['x', 'y'])

df.plot.scatter(x='x', y='y')
plt.xlabel('徒歩時間(分)')
plt.ylabel('家賃(万)/月')
plt.ylim(0, 10)
plt.grid(True)
plt.show()

# 相関係数
df.corr()

# 回帰直線
model = smf.ols(formula = 'y ~ x', data = df)
result = model.fit()
result.summary()

#%%
data = np.array([35, 32, 38, 41, 29, 34, 31, 32, 34])

mean = data.mean()
std = data.std(ddof=1)

t_min, t_max = ss.t(8).interval(0.95)

x_min = mean - 1.96*np.sqrt(11/9)
x_max = mean + 1.96*np.sqrt(11/9)
print('信頼区間1\n', x_min, x_max)

x_min = mean - t_max*std/3
x_max = mean - t_min*std/3
print('信頼区間2\n', x_min, x_max)


#%%
data = np.array([83, 82, 91, 87, 84, 83, 81, 88, 86])
data2 = np.array([i-86 for i in data])


n = len(data)
var = data2 @ data2 / (n-1)
var2 = data.var(ddof=1)
chi_min, chi_max = ss.chi2(9).interval(0.95)

x_min = (n-1)*var / chi_max
x_max = (n-1)*var / chi_min
print('信頼区間1\n', x_min, x_max)

chi_min, chi_max = ss.chi2(8).interval(0.95)
x_min = (n-1)*var2 / chi_max
x_max = (n-1)*var2 / chi_min
print('信頼区間1\n', x_min, x_max)


#%%
(7/10) - 1.96*np.sqrt(0.21/200)
(7/10) + 1.96*np.sqrt(0.21/200)

ss.norm(0, 1).ppf(0.99)


#%%
data = np.array([84, 74, 93, 55, 65, 72, 58, 43, 75, 81])
data.mean()
18*1.64/np.sqrt(10)
s2 = data.var(ddof=1)
t = ss.t(9).ppf(0.95)
60 + t*np.sqrt(s2/10)


#%%
data = np.array([7.88, 6.92, 7.53, 6.56, 7.13, 7.21])
n = len(data)
m = data.mean()
s2 = data.var(ddof=1)

data2 = np.array([i-7.01 for i in data])
s2 = data2 @ data2 / (n-1)

ss.chi2(6).ppf(0.05)
f = (n-1)*s2/0.98
f

ss.chi2(5).ppf(0.05)
f2 = (n-1)*s2/0.98
f2


#%%
15 / np.sqrt((250/13)+(300/9))
Sxy2 = (290*12 + 350*8) / 20
Sxy2

ss.t(20).interval(0.95)
15 / np.sqrt(314*((1/13)+(1/9)))


#%%
data_a = np.array([35, 47, 29, 43, 41])
data_b = np.array([44, 32, 53, 64, 50, 46, 40])

data_a.var(ddof=1) / data_b.var(ddof=1)
ss.f(4, 6).ppf(0.05)
