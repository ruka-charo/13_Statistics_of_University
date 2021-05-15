'''母平均の差の検定'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

# １から作るのは面倒なので関数を作成

# 母分散の値はわからないが、等しいことはわかっている時
def t_dis1(nx, x_mean, x_var, ny, y_mean, y_var):
    s_xy2 = ((nx-1)*x_var + (ny-1)*y_var) / (nx + ny - 2)
    T = (x_mean - y_mean) / np.sqrt(s_xy2 * ((1/nx) + (1/ny)))
    print(T)

# 母分散の値がわからず、等しいこともわからない時
def t_dis2(nx, x_mean, x_var, ny, y_mean, y_var):
    T = (x_mean - y_mean) / np.sqrt((x_var/(nx)) + (y_var/(ny)))

    a = ((x_var/nx) + (y_var/ny))**2
    b = (x_var/nx)**2 / (nx-1)
    c = (y_var/ny)**2 / (ny-1)
    g = a / (b+c)

    print('T=', T)
    print('Tが従うt分布の自由度=', g, 'に一番近い整数')

t_dis2(13, 70, 290, 9, 55, 350)
t_dis2(8, 7.71, 0.21, 11, 8.12, 0.29)
