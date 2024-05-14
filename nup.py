import control 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

s = control.tf('s')

# Tente ler o arquivo com pandas
try:
    df = pd.read_excel('FO.xlsx')
except Exception as e:
    print("Erro ao ler o arquivo Excel:", e)
    df = pd.DataFrame()

# Verifique se o DataFrame foi criado com sucesso
if not df.empty:
    df.describe()

    max_t = df['t'].max()

    delta_t = df.loc[1, 't'] - df.loc[0, 't']

    count_t = len(df['t'])

    def fun(x):
        global df

        kp = x[0]
        tau = x[1]
        sys = kp / (tau*s + 1)

        t_fun, y_fun = control.step_response(sys, T=max_t, T_num=count_t)
        df_fun = pd.DataFrame({'t_fun': t_fun, 'y_fun': y_fun})

        mse = mean_squared_error(df_fun['y_fun'], df['y'])
        return mse

    xo = [1, 1]
    res = minimize(fun, xo, method='powell', tol=1e-6)
    kp_opt = res.x[0]
    tau_opt = res.x[1]
    sys_opt = kp_opt / (tau_opt*s + 1)

    print('kp=', kp_opt)
    print('tau=', tau_opt)

    t_opt, y_opt = control.step_response(sys_opt, T=max_t, T_num=count_t)

    plt.plot(t_opt, y_opt)
    plt.plot(df['t'], df['y'], marker=".", color="red")
    plt.xlabel('time(s)')
    plt.ylabel('output')
    plt.grid()
    plt.show()
else:
    print("Não foi possível ler o arquivo Excel.")
