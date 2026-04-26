# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Training Hamiltonian Neural Networks (HNNs) for Bayesian inference problems
# Original authors of HNNs code: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)
# Available at https://github.com/greydanus/hamiltonian-nn under the Apache License 2.0
# Modified by Som Dhulipala at Idaho National Laboratory for Bayesian inference problems
# Modifications include:
# - Generalizing the code to any number of dimensions
# - Introduce latent parameters to HNNs to improve expressivity
# - Reliance on the leap frog integrator for improved dynamics stability
# - Obtain the training from probability distribution space
# - Use a deep HNN arichtecture to improve predictive performance

import autograd.numpy as np
import torch, argparse
from hnnmc.get_args import get_args
args = get_args()

def functions(coords):
    #******** 1D Gaussian Mixture #********
    if (args.dist_name == '1D_Gauss_mix'):
        q, p = np.split(coords,2)
        mu1 = 1.0
        mu2 = -1.0
        sigma = 0.35
        term1 = -np.log(0.5*(np.exp(-(q-mu1)**2/(2*sigma**2)))+0.5*(np.exp(-(q-mu2)**2/(2*sigma**2))))
        H = term1 + p**2/2 # Normal PDF

    # #******** 2D Gaussian Four Mixtures #********
    elif(args.dist_name == '2D_Gauss_mix'):
        q1, q2, p1, p2 = np.split(coords,4)
        sigma_inv = np.array([[1.,0.],[0.,1.]])
        term1 = 0.
        
        mu = np.array([3.,0.])
        y = np.array([q1-mu[0],q2-mu[1]])
        tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
        term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        mu = np.array([-3.,0.])
        y = np.array([q1-mu[0],q2-mu[1]])
        tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
        term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        mu = np.array([0.,3.])
        y = np.array([q1-mu[0],q2-mu[1]])
        tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
        term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        mu = np.array([0.,-3.])
        y = np.array([q1-mu[0],q2-mu[1]])
        tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
        term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        term1 = -np.log(term1)
        term2 = p1**2/2+p2**2/2
        H = term1 + term2

    # ******** 5D Ill-Conditioned Gaussian #********
    elif(args.dist_name == '5D_illconditioned_Gaussian'):
        dic1 = np.split(coords,args.input_dim)
        var1 = np.array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])
        term1 = dic1[0]**2/(2*var1[0])
        for ii in np.arange(1,5,1):
            term1 = term1 + dic1[ii]**2/(2*var1[ii])
        term2 = dic1[5]**2/2
        for ii in np.arange(6,10,1):
            term2 = term2 + dic1[ii]**2/2
        H = term1 + term2

    # ******** nD Funnel #********
    elif(args.dist_name == 'nD_Funnel'):
        dic1 = np.split(coords,args.input_dim)
        term1 = dic1[0]**2/(2*3**2)
        for ii in np.arange(1,int(args.input_dim/2),1):
            term1 = term1 + dic1[ii]**2/(2 * (2.718281828459045**(dic1[0] / 2))**2)
        term2 = 0.0
        for ii in np.arange(int(args.input_dim/2),args.input_dim,1):
            term2 = term2 + dic1[ii]**2/2 
        H = term1 + term2
    
    # ******** nD Rosenbrock #********
    elif(args.dist_name == 'nD_Rosenbrock'):
        dic1 = np.split(coords,args.input_dim)
        term1 = 0.0
        for ii in np.arange(0,int(args.input_dim/2)-1,1):
            term1 = term1 + (100.0 * (dic1[ii+1] - dic1[ii]**2)**2 + (1 - dic1[ii])**2) / 20.0
        term2 = 0.0
        for ii in np.arange(int(args.input_dim/2),args.input_dim,1):
            term2 = term2 + 1*dic1[ii]**2/2
        H = term1 + term2

    # ******** nD standard Gaussian #********
    elif(args.dist_name == 'nD_standard_Gaussian'):
        dic1 = np.split(coords,args.input_dim)
        var1 = np.ones(int(args.input_dim))
        term1 = dic1[0]**2/(2*var1[0])
        for ii in np.arange(1,int(args.input_dim/2),1):
            term1 = term1 + dic1[ii]**2/(2*var1[ii])
        term2 = 0.0
        for ii in np.arange(int(args.input_dim/2),args.input_dim,1):
            term2 = term2 + dic1[ii]**2/2
        H = term1 + term2
    # ******** 2D Banana shaped #********
    elif (args.dist_name == 'banana'):
        # coords: [q1, q2, p1, p2]  —— 确保 args.input_dim == 4
        q1, q2, p1, p2 = np.split(coords, 4)

        # banana 变换参数（与论文示例一致）
        a   = 1.15
        b   = 0.5
        rho = 0.9

        # 注意：Sigma_inv 必须对称（-rho）
        Sigma_inv = (1.0/(1.0 - rho**2)) * np.array([[ 1.0, -rho],
                                                      [-rho,  1.0]])
        # u(q)
        u1 = a * q1
        u2 = a * (q2 - b * q1**2)

        # U(q) = 0.5 * u^T Σ^{-1} u
        # 这里是标量：u=[u1,u2]
        quad = (u1*Sigma_inv[0,0] + u2*Sigma_inv[0,1]) * u1 \
             + (u1*Sigma_inv[1,0] + u2*Sigma_inv[1,1]) * u2
        U = 0.5 * quad

        # K(p) = 0.5 * (p1^2 + p2^2)
        K = 0.5 * (p1**2 + p2**2)

        H = U + K
    # ******** annulus (truncated standard Gaussian via soft barrier) ********
    elif (args.dist_name == 'annulus'):
        # y = [q1, q2, p1, p2]
        q1, q2, p1, p2 = np.split(coords, 4)
        # 几何参数（你可放 get_args 里统一配置）
        R_IN, R_OUT = 3.0, 5.0
        LAMBDA = 200.0      # 障壁强度（越大越贴近硬约束）
        MARGIN = 0.0        # 如需让环稍微“厚一点”做缓冲，可设 0.1~0.3

        r = np.sqrt(q1**2 + q2**2)
        relu = lambda z: np.maximum(0.0, z)
        # 内外两侧惩罚（可微）：越出环即二次罚
        pen = LAMBDA * (relu(R_IN - MARGIN - r)**2 + relu(r - (R_OUT + MARGIN))**2)

        U = 0.5*(q1**2 + q2**2) + pen      # 势能 = 标准高斯 + 软障壁
        K = 0.5*(p1**2 + p2**2)            # 动能
        H = U + K
    else:
        raise ValueError("probability distribution name not recognized")

    return H