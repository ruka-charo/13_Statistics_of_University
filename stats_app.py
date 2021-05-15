import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import tkinter as tk



#%% データセット
x_list = np.array([1.2, 0.8, 1.5, 1.2, 1.0])
y_list = np.array([0.8, 0.4, 0.6, 1.0, 0.5, 0.4, 0.5])

'''変数'''
nx = len(x_list)            #xのデータ数
ny = len(y_list)            #yのデータ数
x_mean = x_list.mean()      #xの標本平均
y_mean = y_list.mean()      #yの標本平均
x_var = x_list.var(ddof=1)  #xの不偏分散
y_var = y_list.var(ddof=1)  #yの不偏分散
s_xy2 = ((nx-1)*x_var + (ny-1)*y_var) / (nx + ny - 2)

#%%

'''正規分布'''
# 母平均の分析
def norm_dis(x_list, alpha=0.05, z=0):
    print('---正規分布による分析---')
    # t分布の有意水準に基づく最大値と最小値
    n_min, n_max = ss.norm(loc=0, scale=1).interval(1 - alpha)
    # 母平均の信頼区間
    x_min = x_mean - n_min*np.sqrt(x_var / nx)
    x_max = x_mean + n_max*np.sqrt(x_var / nx)
    # 信頼区間の表示
    print('母平均の' + str(100*(1-alpha)) + '%信頼区間\n', x_min, '< μ <', x_max)
    # 仮説の検定
    if z != 0:
        print('帰無仮説H0：μ =', z)
        if x_min <= z <= x_max:
            print('帰無仮説H0：μ =', z, 'は棄却されない。')
        else:
            print('帰無仮説H0：μ =', z, 'は棄却。')


'''t分布'''
# 母平均の分析
def t_dis(x_list, alpha=0.05, z=0):
    print('---t分布による分析---')
    # t分布の有意水準に基づく最大値と最小値
    t_min, t_max = ss.t(nx - 1).interval(1 - alpha)
    # 母平均の信頼区間
    x_min = x_mean + t_min*np.sqrt(x_var / nx)
    x_max = x_mean + t_max*np.sqrt(x_var / nx)
    # 信頼区間の表示
    print('母平均の' + str(100*(1-alpha)) + '%信頼区間\n', x_min, '< μ <', x_max)
    # 仮説の検定
    if z != 0:
        print('帰無仮説H0：μ =', z)
        if x_min <= z <= x_max:
            print('帰無仮説H0：μ =', z, 'は棄却されない。')
        else:
            print('帰無仮説H0：μ =', z, 'は棄却。')


# 母平均の差の分析
def t_dis2(x_list, y_list, alpha=0.05):
    print('---t分布による分析---')
    # t分布の有意水準に基づく最大値と最小値
    t_min, t_max = ss.t(nx + ny - 2).interval(1 - alpha)
    # 母平均の信頼区間
    x_min = t_min * np.sqrt(s_xy2 * ((1/nx) + (1/ny)))
    x_max = t_max * np.sqrt(s_xy2 * ((1/nx) + (1/ny)))
    # 信頼区間の表示
    print('母平均の差の' + str(100*(1-alpha)) + '%信頼区間\n', x_min, '< x - y <', x_max)
    print('標本平均の差：', x_mean - y_mean)
    # 仮説の検定
    print('帰無仮説H0：μx = μy')
    if x_min <= (x_mean - y_mean) <= x_max:
        print('帰無仮説H0：μx = μyは棄却されない。')
    else:
        print('帰無仮説H0：μx = μyは棄却。')


'''f分布'''
def f_dis(x_list, y_list, alpha=0.05, z=0):


'''chi2分布'''
# 母分散の分析
def chi2_dis(x_list, alpha=0.05, z=0):
    print('---χ^2分布による分析---')
    # t分布の有意水準に基づく最大値と最小値
    χ_min, χ_max = ss.chi2(nx - 1).interval(1 - alpha)
    # 母平均の信頼区間
    x_min = ((nx - 1)*x_var) / χ_max
    x_max = ((nx - 1)*x_var) / χ_min
    # 信頼区間の表示
    print('母分散の' + str(100*(1-alpha)) + '%信頼区間\n', x_min, '< σ^2 <', x_max)
    # 仮説の検定
    if z != 0:
        print('帰無仮説H0：σ^2 =', z)
        if x_min <= z <= x_max:
            print('帰無仮説H0：μ = ', z, '  は棄却されない。')
        else:
            print('帰無仮説H0：μ =', z, 'は棄却。')
