# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

#x[0] -> k, x[1] -> r, x[2] -> tm, x[3] -> a
def fun(coefs, time, y_value):
    """ Calculates the adjustment error (richards with the provided coefficients minus the real value).
    Keyword arguments:
    coefs -- the initial guess for the coefficiants
    time -- the time variable
    yvalue -- expected value at the specified time to calculate the error for optimization
    """
    return (coefs[0]/(1+np.exp(-coefs[1] * (time - coefs[2])))**(1 / coefs[3])) - y_value

def fun2(coefs, time):
    """ Implements the Richards function. """
    return coefs[0]/(1+np.exp(-coefs[1] * (time - coefs[2])))**(1 / coefs[3])

f_path = '/Users/emanoel/Dropbox/UFPE/Doutorado/Dados/github/ghtorrent/mathematica/programadores/mes/dados_Java_mes.dat'
types = np.dtype([('month', np.int32), ('infected', np.int32)])
x, y = np.loadtxt(f_path, dtype=types, unpack=True)

initial_coefs = np.array([y[-1], 0.01, 50, 0.1])

res_robust = least_squares(fun, initial_coefs, args=(x, y))

x_part = x[:x.size*0.9]
y_part = y[:y.size*0.9]
res_robust_part = least_squares(fun, initial_coefs, args=(x_part, y_part))
residuals = np.array([fun(res_robust.x,x_val, y_val) for x_val, y_val in zip(x,y)])

"""
plt.plot(x, y, '^', label='data')
plt.plot(x, fun2(res_robust.x, x), label='fit full')
plt.plot(x, fun2(res_robust_part.x, x), label='fit part')
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.legend()
plt.show()
"""

#plt.clf()
plt.plot(x, fun(res_robust.x,x, y), 'o',label='residual')
plt.xlabel('$t$')
plt.ylabel('$residual$')
plt.grid(True, axis='both')
plt.vlines(x=x, ymin=np.zeros(residuals.size), ymax=residuals, color='gray')
plt.legend()
plt.show()