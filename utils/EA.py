'''
文件用途：
作者：陈欣如
日期：2022年12月07日
'''
import numpy as np
import scipy
from scipy.linalg import fractional_matrix_power
# def EA(X, EA_R, EA_Num, EARstack, EANumstack, pids):
#     num_trial, num_channel, num_sampls = np.shape(X)
#     R = np.zeros((num_channel, num_channel))
#     for i in range(num_trial):
#         XTemp = np.squeeze(X[i, :, :])
#         R = R + np.dot(XTemp, XTemp.T)
#     if EA_Num == 0:
#         EA_Num = num_trial
#         EA_R = R
#     else:
#         EA_Num = EA_Num + num_trial
#         EA_R = EA_R + R
#     if EA_Num < 200:# and pids[-1] == pids[-2]:
#         R = (EA_R + EARstack) / (EA_Num + EANumstack)
#     else:
#         R = (EA_R) / (EA_Num)
#     R = fuerfenzhiyi(R)
#     for i in range(num_trial):
#         XTemp = np.squeeze(X[i, :, :])
#         XTemp = np.dot(R, XTemp)
#         X[i, :, :] = XTemp
#     # print(EA_R)
#     return X, EA_R, EA_Num


def fuerfenzhiyi(R):
    v, Q = np.linalg.eig(R)
    ss1 = np.diag(v ** (-0.5))
    ss1[np.isnan(ss1)] = 0
    re = np.dot(Q, np.dot(ss1, np.linalg.inv(Q)))
    return np.real(re)

def EA( x): #x(bs,channel,point)
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1])) #(bs,channel,channel)
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (0.00000001) * np.eye(x.shape[1])
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA,sqrtRefEA