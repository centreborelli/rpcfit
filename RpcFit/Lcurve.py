#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:08:19 2020

@author: rakiki
"""
from scipy import interpolate,optimize,linalg
import numpy as np
import matplotlib.pyplot as plt

### Standard tikhonov Lcurve functions
def curvature(lambd, sig, beta, xi, rhoLS2):
    '''
    Computes the curvature.(-1) of the standard tikhonov Lcurve
    Args: 
        lambd: 1D np array of regularization parameter values
        sig: 1D np array of singular values for design matrix
        beta: projection of right equation term onto left singular vectors
        xi: beta / sig with safedivision
        rhoLS2: the orthogonal residual 
    Returns : 
        The negative of the curvature ( to minimize it instead of maximize)
    '''
    # Initialization.
    phi = np.zeros(lambd.shape)
    dphi = np.zeros(lambd.shape)
    psi = np.zeros(lambd.shape)
    dpsi = np.zeros(lambd.shape)
    eta = np.zeros(lambd.shape)
    rho = np.zeros(lambd.shape)

    # Compute some intermediate quantities.
    for jl, lam in enumerate(lambd):
        f  = np.divide((sig ** 2), (sig ** 2 + lam ** 2)) 
        cf = 1 - f 
        eta[jl] = np.linalg.norm(f * xi) 
        rho[jl] = np.linalg.norm(cf * beta)
        f1 = -2 * f * cf / lam 
        f2 = -f1 * (3 - 4*f)/lam
        phi[jl]  = np.sum(f*f1*np.abs(xi)**2) 
        psi[jl] = np.sum(cf*f1*np.abs(beta)**2)
        dphi[jl] = np.sum((f1**2 + f*f2)*np.abs(xi)**2)
        dpsi[jl] = np.sum((-f1**2 + cf*f2)*np.abs(beta)**2) 
    rho = np.sqrt(rho ** 2 + rhoLS2)

    # Now compute the first and second derivatives of eta and rho
    # with respect to lambda;
    deta  =  np.divide(phi, eta) 
    drho  = -np.divide(psi, rho)
    ddeta =  np.divide(dphi, eta) - deta * np.divide(deta, eta)
    ddrho = -np.divide(dpsi, rho) - drho * np.divide(drho, rho)

    # Convert to derivatives of log(eta) and log(rho).
    dlogeta  = np.divide(deta, eta)
    dlogrho  = np.divide(drho, rho)
    ddlogeta = np.divide(ddeta, eta) - (dlogeta)**2
    ddlogrho = np.divide(ddrho, rho) - (dlogrho)**2
    # curvature.
    curv = - np.divide((dlogrho * ddlogeta - ddlogrho * dlogeta),
        (dlogrho**2 + dlogeta**2)**(1.5))
    return curv

def safedivision(num, denum):
    '''
    Computes the elementwise division of vec1 by vec2
    if vec2[i] = 0, result[i] = 0
    Args: 
        num, denum: np.array
    Returns: 
        num / denum, with 0 where denum = 0
    '''
    denum = denum.reshape(num.shape)
    result = np.zeros_like(num)
    mask = np.where(denum!= 0)
    result[mask] = num[mask]/denum[mask]
    return result

def l_corner(reg_param,u,sig,bm, plotit = False):
    '''
    computes the corner of the L-curve for standard tikhonov
    Inputs:
        reg_param - computed in l_curve function
        u left side matrix computed from svd (size: Nm x Nu) 
        sig is the singular value vector of A
        bm is the measured results
        plotit : whether to plot the curvature
   Returns: 
        reg_c: best regularization parameter, not necessarily in reg_param
        curv_id: id of regularization parameter closest to reg_c
    '''
    # Set default parameters for treatment of discrete L-curve.
    Nm, Nu = u.shape
    npoints = len(reg_param)
    beta = (u.T @ bm).reshape(-1,1)
    # ortho resid
    b0 = bm - u.dot(beta)
    rhoLS2 = np.linalg.norm(b0)**2
    beta = beta.ravel()
    xi = safedivision(beta, sig)
    # Call curvature calculator
    curv = curvature(reg_param, sig, beta, xi, rhoLS2) 
    # Minimize on interval
    curv_id = np.argmin(curv)
    x1 = reg_param[int(np.amin([curv_id + 1, npoints - 1 ]))]
    x2 = reg_param[int(np.amax([curv_id - 1, 0]))]
    if plotit: 
        # TODO fix plotting
        plt.figure()
        plt.title('curvature')
        plt.xlabel('regularization')
        plt.ylabel('curvature')
        plt.plot(reg_param, -curv, marker = '.')
        plt.scatter(reg_param[curv_id], -curv[curv_id], marker = 'x', color = 'r')
        plt.show()
    reg_c = optimize.fminbound(curvature, x1, x2, args = (sig, beta, xi, rhoLS2), full_output=False, disp=False)
    kappa_max = - curvature(reg_c, sig, beta, xi, rhoLS2)# Maximum curvature.
    if kappa_max[0] < 0:
        # in case the curvature is always negative ( concave )
        # take values for the smallest regularization param
        reg_c = reg_param[-1]
    return reg_c[0], curv_id
    

def l_curve(A, bm, plotit = False):
    '''
    Plot the L-curve and find its "corner" for standard tikhonov 
    Inputs:
        A: design matrix, is of size Nm x Nu,
        where Nm are the number of measurements and Nu the number of unknowns
        bm: your measurement vector (size: Nm x 1)
        plotit: plot the l curve and the curvature
    Returns: 
        lam_opt: the optimal regularization parameter
    '''
    u, sig , v = linalg.svd(A, full_matrices=False)
    # Set defaults.
    npoints = 200  # Number of points on the L-curve
    smin_ratio = 16*np.finfo(float).eps  # Smallest regularization parameter.
    # Initialization.
    Nm, Nu = u.shape
    beta = (u.T @ bm ).reshape(-1,1)
    b0 = bm - u.dot(beta)
    rhoLS2 = np.linalg.norm(b0)**2
    beta = beta.ravel()
    xi = safedivision(beta, sig)

    eta = np.zeros((npoints,1))
    rho = np.zeros((npoints,1)) 
    reg_param = np.zeros((npoints,1))
    s2 = sig ** 2
    # reg_param list from smallest singular val to biggest
    reg_param[-1] = np.amax([sig[-1], sig[0]*smin_ratio])
    ratio = (sig[0]/reg_param[-1]) ** (1/(npoints-1))
    for i in np.arange(start=npoints-2, step=-1, stop = -1):
        reg_param[i] = ratio*reg_param[i+1]
    for i in np.arange(start=0, step=1, stop = npoints):
        f = s2 / (s2 + reg_param[i] ** 2) # all filter coefs
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm((1-f) * beta)
    if (Nm > Nu):
        rho = np.sqrt(rho ** 2 + rhoLS2)
    # Compute the corner of the L-curve (optimal regularization parameter)
    lam_opt, c_id = l_corner(reg_param,u,sig,bm, plotit)
    # Compute the optimal solution 
    f = s2 / (s2 + lam_opt** 2) # all filter coefs
    sol_opt = np.sum((f * xi).reshape(-1,1) * v, axis = 0).reshape(-1,1)
    # want to plot the L curve?
    if plotit:
        # TODO fix plotting
        plt.figure()
        plt.loglog(rho ,eta , marker = '.')
        plt.scatter(rho[c_id],eta[c_id], marker = 'X', color = 'r')
        plt.xlabel('Residual norm ||Ax - b||')
        plt.ylabel('Solution norm ||x||')
        plt.show()

    return lam_opt, sol_opt


def LSQ_iter(A, l, h):
    '''
    returns the lsq solution and norm, and resid norm of
    min || AJ - l ||**2 + h**2 ||J||**2
    with svd decomposition
    '''
    u, sig, v = linalg.svd(A, full_matrices=False)
    bm = l
    # Initialization.
    Nm, Nu = u.shape
    beta = (u.T @ bm ).reshape(-1,1)
    b0 = bm - u.dot(beta)
    rhoLS2 = np.linalg.norm(b0)**2
    beta = beta.ravel()
    xi = safedivision(beta, sig)
    reg_param = h
    s2 = sig ** 2
    f = s2 / (s2 + reg_param ** 2) # all filter coefs
    solution = np.sum((f * xi).reshape(-1,1) * v, axis = 0).reshape(-1,1)
    eta = np.linalg.norm(f * xi)
    rho = np.linalg.norm((1-f) * beta)
    if (Nm > Nu ):
        rho = np.sqrt(rho ** 2 + rhoLS2)
    return rho, eta, solution

def LSQ_iter_bis(X, y, h):
    '''
    returns the lsq solution and norm, and resid norm of
    min || AJ - l ||**2 + h**2 ||J||**2
    with svd decomposition
    Adapted from sklearn
    '''
    U, s, Vt = linalg.svd(X, full_matrices=False)
    idx = s > 1e-15  # same default value as scipy.linalg.pinv
    s_nnz = s[idx][:, np.newaxis]
    UTy = np.dot(U.T, y)
    d = np.zeros((s.size,1), dtype=X.dtype)
    d[idx] = s_nnz / (s_nnz ** 2 + h**2)
    d_UT_y = d * UTy
    sol =  np.dot(Vt.T, d_UT_y)
    residue = np.linalg.norm(X.dot(sol) - y)
    norm = np.linalg.norm(sol)
    return residue, norm, sol


def get_reg_param(X, npoints, truncate = False):
    '''
    Generates the list of regularization parameters on which to evaluate 
    the L curve
    Args: 
        X: the design matrix of the LSQ problem
        npoints: the number of points on the lcurve, ie the length of the 
                regularization parameters list
        truncate: if False, regularization goes from sig_min to sig_max in a geometric manner
                    if True, we just consider the principal singular value ( is not really stable)
    Returns: 
        The regularization parameters list
    '''
    reg_param = np.zeros((npoints,1))
    _, sig, _ = linalg.svd(X, full_matrices=False)
    smin_ratio = 100*np.finfo(float).eps 
    #TODO check if this strategy is always valid
    if truncate:
        smax = sig[np.argmax(sig[:-1]/sig[1:]) + 1]
    else: 
        smax = sig[0]
    # smallest regularization parameter
    reg_param[-1] = np.amax([sig[-1], smax*smin_ratio])
    ratio = (smax/reg_param[-1]) ** (1/(npoints-1))
    for i in np.arange(start=npoints-2, step=-1, stop = -1):
        reg_param[i] = ratio*reg_param[i+1]
    return reg_param

def find_corner(eta, rho, plot = False):
    '''
    Finds the corner of the Lcurve, based on spline interpolation
    Args: 
        eta: the list of solution norms
        rho: the list of residu norms
        plot: Whether to plot the Lcurve and the curvature
    Returns: 
        id of the corner of the Lcurve
    '''
    deg = 2 # degree of local smoothing polynomial
    q = 2 # half-width of local smoothing interval
    order = 3# order of fitting 2D spline curve
    # convert to logarithms
    lrho = np.log(rho)
    leta = np.log(eta)
    # filter points on Lcurve based on solution norm     
    lr = len(lrho)
    
    # smoothing 
    slrho = np.copy(lrho) 
    sleta = np.copy(leta)
    # for all interior points  q, lr-1-q, perform local smoothing 
    # with polynomial degree deg to k-q:k+q
    v = np.arange(-q, q+1)
    A = np.zeros((2*q + 1, deg + 1))
    A[:, 0] = np.ones((2*q + 1,))
    for j in range(1, deg + 1):
        A[:,j] = A[:,j-1] * v
    Ainv = np.linalg.pinv(A)
    for k in range(q, lr - q):
        slrho[k] = Ainv[0].dot(lrho[k + v])
        sleta[k] = Ainv[0].dot(leta[k + v])
    
    # fit a 2D spline curve to the smoothed discrete L-curve
    tck, u = interpolate.splprep([slrho.ravel(), sleta.ravel()], s = 0, k = order)
    spl_pts = 200 # number of pts on which to evaluate curvature 
    while(len(u)<spl_pts):
        # include u values at the middle of previous knots
        u = np.sort(np.hstack( [u , (u[1:] + u[:-1])/2]))
    xy = interpolate.splev(u, tck, der = 0)
    D = interpolate.splev(u, tck, der = 1)
    DD = interpolate.splev(u, tck, der = 2)
    # compute the corner of the discretized spline curve 
    # via max curvature
    k1 = D[0] * DD[1] - DD[0]*D[1] 
    k2 = (D[0]**2 + D[1]**2)**(1.5)
    kappa = safedivision(-k1, k2)
    # cut indices where monotonicity is not satisfied 
    id_max = len(kappa)
    min1 = np.where(D[1] < 0 )[0]
    min1 = min1[0] if len(min1) else id_max
    min2 = np.where(D[0] > 0 )[0]
    min2 = min2[0] if len(min2) else id_max
    cut_id = min( min1, min2 )  
    
    ikmax = np.argmax(kappa[:cut_id])
    x_corner, y_corner = interpolate.splev(u[ikmax], tck)
    assert x_corner == xy[0][ikmax]
    assert y_corner == xy[1][ikmax]
    kmax = kappa[ikmax]
    if kmax < 0:
            # then just take the smallest regularization parameter
            cid = -1
    else: 
    
            index = np.where(np.logical_and(lrho < x_corner,leta<y_corner))
            if len(index[0]) > 0:
                # restrict search values
                cid =  np.argmin((lrho[index] - x_corner)**2 + (leta[index]  - y_corner)**2)
                cid = index[0][cid]
            else: 
                # brute force search 
                cid = np.argmin((lrho - x_corner)**2 + (leta - y_corner)**2)
    if plot: 
        plt.figure() 
        plt.plot(xy[0][:cut_id], kappa[:cut_id], marker = '.')
        plt.scatter( xy[0][ikmax], kappa[ikmax], marker='v', color ='r')
        plt.xlabel('norm of residu ')
        plt.ylabel('curvature')
        plt.title('curvature')
        plt.show()
        plt.figure()
        plt.plot(xy[0][:cut_id], xy[1][:cut_id], '.', color = 'red', label= 'spline')
        plt.plot(lrho,leta, '.', label = 'logcoord')
        plt.scatter(lrho[cid], leta[cid], marker = 'x', color = 'black')
        plt.legend()
        plt.xlabel('norm of residu')
        plt.ylabel('norm of solution')
        plt.title('L - curve') 
        plt.show()
    return cid

