# -*- coding: utf-8 -*-
"""
@author: PengChuan
    评估的log
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def read_loss_stoi_from_log(log_path):
    def get_arg(str_line, str_arg):
        if str_arg not in str_line:
            return 0.
        s = str_line.split(str_arg)[1]
        s = s.split()[0].replace(',', '')
        return float(s)
    d_cost, d_acc, g_cost, g_mse, _stoi = [], [], [], [], []
    with open(log_path, 'r') as fd:
        for line in fd:
            if "d_cost=" in line:
                d_cost.append(get_arg(line, 'd_cost='))
                d_acc.append(get_arg(line, 'd_acc='))
                g_cost.append(get_arg(line, 'g_cost='))
                g_mse.append(get_arg(line, 'g_mse='))
                _stoi.append(get_arg(line, 'eval='))

    return [d_cost, d_acc, g_cost, g_mse, _stoi]

if __name__ == '__main__':
    log_path = 'zlog/loss/prt_wgan_loss-D.log'
    print(log_path)

    loss_stoi = read_loss_stoi_from_log(log_path)
    loss_stoi = np.array(loss_stoi)
    corrcoef = np.corrcoef(loss_stoi)
    print(corrcoef)

    loss_stoi = loss_stoi.T
    linreg = LinearRegression()
    model = linreg.fit(loss_stoi[:, [0, 2]], loss_stoi[:, 4])
    print(linreg.intercept_)
    print(linreg.coef_)
    # print(loss_stoi[:, 2:5])
