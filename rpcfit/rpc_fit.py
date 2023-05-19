from typing import Optional
import numpy as np
from rpcfit import Lcurve
from rpcm import rpc_model 

def poly_vect(x, y, z):
    """
    Returns evaluated polynomial vector without the first constant term equal to 1,
    using the order convention defined in rpc_model.apply_poly
    """
    return np.array([y, x, z, 
                     y*x, y*z, x*z, y*y, x*x, z*z, 
                     x*y*z, y*y*y, y*x*x, y*z*z, y*y*x, x*x*x, x*z*z, y*y*z, x*x*z, z*z*z])

def normalize_target(rpc, target):
    """
    Normalize in image space
    """
    target_norm = np.vstack(((target[:, 0] - rpc.col_offset) / rpc.col_scale,
                             (target[:, 1] - rpc.row_offset) / rpc.row_scale)).T
    return target_norm

def normalize_input_locs(rpc, input_locs):
    """
    Normalize in world space
    """
    input_locs_norm = np.vstack(((input_locs[:, 0] - rpc.lon_offset) / rpc.lon_scale,
                                 (input_locs[:, 1] - rpc.lat_offset) / rpc.lat_scale,
                                 (input_locs[:, 2] - rpc.alt_offset) / rpc.alt_scale)).T
    return input_locs_norm    

def update_rpc(rpc, x, orientation = "projection"):
    """
    Update rpc coefficients
    orientation: if "projection" assume the coefficients are for row and col
    otherwise for lon and lat
    """
    if orientation == "projection": 
        rpc.row_num, rpc.row_den = x[:20], x[20:40]
        rpc.col_num, rpc.col_den = x[40:60], x[60:]
    else: 
        rpc.lat_num, rpc.lat_den = x[:20], x[20:40]
        rpc.lon_num, rpc.lon_den = x[40:60], x[60:]
    return rpc

def calculate_RMSE_row_col(rpc, input_locs, target):
    """
    Calculate MSE & RMSE in image domain
    """
    col_pred, row_pred = rpc.projection(lon=input_locs[:,0], lat=input_locs[:,1], alt=input_locs[:,2])
    MSE_col, MSE_row = np.mean((np.hstack([col_pred.reshape(-1, 1), row_pred.reshape(-1, 1)]) - target) ** 2, axis=0)
    MSE_row_col = np.mean([MSE_col, MSE_row]) # the number of data is equal in MSE_col and MSE_row
    RMSE_row_col = np.sqrt(MSE_row_col)
    return RMSE_row_col
def evaluate(rpc, input_locs, target):
    '''
    Args: 
    rpc: RPCModel obj
    input_locs: Nx3 (lon, lat, alt)
    target: Nx2 (col, row)
    Returns: 
    RMSE (col, row) 
    max_err tuple (col, row) ,
    planimetry Nx1 (error in px))
    '''
    col_pred, row_pred = rpc.projection(lon=input_locs[:,0], lat=input_locs[:,1], alt=input_locs[:,2])
    error = np.hstack([col_pred.reshape(-1, 1), row_pred.reshape(-1, 1)]) - target
    max_err = np.amax(np.abs(error), axis = 0)
    RMSE = np.sqrt(np.mean(error**2, axis = 0))
    planimetry = np.linalg.norm(error, axis = 1)
    return RMSE , max_err, planimetry

def weighted_lsq(rpc_to_calibrate, target, input_locs, h=1e-3, tol=1e-2, max_iter=20):
    """
    Regularized iterative weighted least squares for calibrating rpc.
    
    Args: 
        max_iter : maximum number of iterations
        h : regularization parameter
        tol : tolerance criterion on improvment of RMSE over iterations
        
    Warning: this code is to be employed with the rpc_model defined in s2p
    """
    reg_matrix = (h ** 2) * np.eye(39)  # regularization matrix
    target_norm = normalize_target(rpc_to_calibrate, target)  # col, row
    input_locs_norm = normalize_input_locs(rpc_to_calibrate, input_locs)  # lon, lat, alt

    # define C, R and M
    C, R = target_norm[:, 0][:, np.newaxis], target_norm[:, 1][:, np.newaxis]
    lon, lat, alt = input_locs_norm[:,0], input_locs_norm[:,1], input_locs_norm[:,2]
#    col, row = target_norm[:,0][:, np.newaxis], target_norm[:,1][:, np.newaxis]
    vect = poly_vect(x=lat, y=lon, z=alt).T
    MC = np.hstack([np.ones((lon.shape[0], 1)), vect, -C * vect])
    MR = np.hstack([np.ones((lon.shape[0], 1)), vect, -R * vect])
       
    # calculate direct solution
    JR = np.linalg.inv(MR.T @ MR) @ (MR.T @ R)
    JC = np.linalg.inv(MC.T @ MC) @ (MC.T @ C)

    # update rpc and get error
    coefs = np.vstack([JR[:20], 1, JR[20:], JC[:20], 1, JC[20:]]).reshape(-1)
    rpc_to_calibrate = update_rpc(rpc_to_calibrate, coefs)
    RMSE_row_col = calculate_RMSE_row_col(rpc_to_calibrate, input_locs, target)
    for n_iter in range(1, max_iter+1):
        WR2 = np.diagflat(1 / ((MR[:, :20] @ coefs[20:40]) ** 2))  # diagonal matrix with 1 / denom ** 2
        JR_iter = np.linalg.inv((MR.T @ WR2 @ MR) + reg_matrix) @ (MR.T @ WR2 @ R)
        WC2 = np.diagflat(1 / ((MC[:, :20] @ coefs[60:80]) ** 2))  # diagonal matrix with 1 / denom ** 2
        JC_iter = np.linalg.inv((MC.T @ WC2 @ MC) + reg_matrix) @ (MC.T @ WC2 @ C)

        # update rpc and get error
        coefs = np.vstack([JR_iter[:20], 1, JR_iter[20:], JC_iter[:20], 1, JC_iter[20:]]).reshape(-1)
        rpc_to_calibrate = update_rpc(rpc_to_calibrate, coefs)
        RMSE_row_col_prev = RMSE_row_col
        RMSE_row_col = calculate_RMSE_row_col(rpc_to_calibrate, input_locs, target)
        # check convergence
        if np.abs(RMSE_row_col_prev - RMSE_row_col) < tol:
            break
    
    return rpc_to_calibrate
    
def get_mats(rpc_to_calibrate, target, input_locs, orientation = "projection"):
    '''
    Computes the design matrices used in the least squares fit
    Args: 
        rpc_to_calibrate: RPCModel instance, should contain normalization coefficients
        input_locs: Nx3 (lon, lat, alt)
        target: Nx2 (col, row)
        orientation: "projection" returns matrices for projection rpc fitting
                     "localization" returns matrices for localization rpc fitting
    Returns: 
        MR, MC, A: design matrices for row, col, joint problem estimation respectively
        R, C, l: the right hand side term for row, col, joint problem estimation respectively

    '''
    target_norm = normalize_target(rpc_to_calibrate, target)  # col, row
    input_locs_norm = normalize_input_locs(rpc_to_calibrate, input_locs)  # lon, lat, alt
    col , row = target_norm[:, 0], target_norm[:, 1]
    lon, lat, alt = input_locs_norm[:, 0], input_locs_norm[:, 1], input_locs_norm[:, 2]
    if orientation == "projection": 
        # define C, R  and x, y, z
        C, R = col[:, np.newaxis], row[:, np.newaxis]
        x, y, z = lat, lon, alt # same order as RPCModel 
    else: 
        # define C, R and x, y, z 
        C, R = lon[:, np.newaxis], lat[:, np.newaxis]
        x, y, z = row, col, alt # same order as RPCModel 
    vect = poly_vect(x=x, y=y, z=z).T
    num_samp = C.shape[0]
    MC = np.hstack([np.ones((num_samp, 1)), vect, -C * vect])
    MR = np.hstack([np.ones((num_samp, 1)), vect, -R * vect])
    A = np.zeros((2 * num_samp, 78))
    A[:num_samp, :39] = MR
    A[num_samp:, 39:] = MC
    l = np.vstack([R, C])
    return MR, R, MC, C, A, l
    
def compute_rmse(X, y, theta, scale = (1,)):
    '''
    computes rmse without rpc model
    Args: 
        X: design mat
        y: right side term
        theta: current solution
        scale: y normalization scale
    Returns: 
        RMSE
    '''
    weights = get_weight(X, theta)
    # check scale
    if theta.size == 39: 
        scaling = scale[0] * np.ones((X.shape[0], 1))
    elif theta.size == 78:
        num_samp = X.shape[0]//2
        scaling = np.vstack([scale[0] * np.ones((num_samp, 1)), 
                             scale[1] * np.ones((num_samp, 1))])
    RMSE = np.sqrt(np.mean((scaling*weights*(X @ theta - y))**2))
    return RMSE, weights

def get_weight(X, theta):
    '''
    Computes the weights from the desing matrix and current solution
    Args: 
        X: design matrix
        theta: current solution
    Returns: 
        weights
    '''
    if theta.size == 39:
        # Seperately solving row and col
        weights = 1/ ( X[:,:20] @ np.vstack([ 1, theta[20:39]]))
    elif theta.size == 78:
        # jointly solving row and col
        num_samp = int(X.shape[0]//2)
        R = 1 / ( X[:num_samp, :20] @ np.vstack([ 1, theta[20:39]]))
        C = 1 / ( X[num_samp:, 39:59] @ np.vstack([ 1, theta[59:78]]) )
        weights = np.vstack([R.reshape(-1,1), C.reshape(-1, 1)])
    return weights.reshape(-1,1)
  
def wlsq_svd(X , y, tol=1e-2, max_iter=20, h = None, plot = False, scale = (1,)):
    '''
    iterative weighted least squares, initial solution and regularization parameter 
    computed by means of Lcurve heuristic, if regularization parameter not given
    Then, joint row and col iterative weighted regularized least squares optim
    based on svd decomposition
    each iteration performs and svd decomposition to find the solution
    Args: 
        X: design matrix
        y: right side term
        max_iter : maximum number of iterations
        tol : tolerance criterion on improvment of RMSE over iterations
        h: regularization parameter, if None, compute by Lcurve
        plot: whether to plot Lcurve, ignore if Lcurve is not used
        scale: scale used to normalize the right side coordinate
                tuple (scale_row, scale_col) when solving jointly, otherwise 
                tuple (scale_row, ) or tuple (scale_col, )
    Returns: 
        solution vector, norm_res, norm_sol
    '''    
    
    if h is None: 
        # first iteration, svd solver
        h, theta = Lcurve.l_curve(X, y, plotit=plot)
        norm_sol_best = np.linalg.norm(theta)
        norm_res_best = np.linalg.norm(X @ theta - y)
    else: 
        norm_res_best, norm_sol_best, theta = Lcurve.LSQ_iter(X, y , h)
    theta = theta.reshape((-1,1))
    # get error
    RMSE, weights = compute_rmse(X, y, theta, scale)
    theta_best = theta
    RMSE_best = RMSE
    
    # weighted least squares
    for n_iter in range(1, max_iter+1):
            # iterate, svd solver
            norm_res , norm_sol , theta_iter = Lcurve.LSQ_iter(weights*X, weights*y , h)
            theta_iter = theta_iter.reshape((-1,1))
            # store previous rmse 
            RMSE_prev = RMSE
            # get rmse and weights
            RMSE, weights = compute_rmse(X, y, theta_iter, scale)
            # store best solution
            if RMSE < RMSE_best: 
                theta_best = theta_iter
                RMSE_best = RMSE
                norm_res_best = norm_res
                norm_sol_best = norm_sol
            # check convergence
            if np.abs(RMSE_prev - RMSE) < tol:
                break
    return theta_best, norm_res_best, norm_sol_best, h

def solve_lc(X, y, tol=1e-2, max_iter=20, method = "initLcurve", plot = False, scale = (1,)):
    '''
    Solves argmin(theta) || X.theta - y ||^2 + h^2 ||theta||^2
    h is the regularization parameter found automatically by Lcurve
    Uses weighted least squares with svd solver
    Args:
        X: design matrix
        y: right side term
        max_iter : maximum number of iterations
        tol : tolerance criterion on improvment of RMSE over iterations
        method: the method of finding the regularization h automatically
                'initLcurve' will do a standard tikhonov Lcurve without weights
                to find the h parameter, uses closed form formulas
                'discreteLcurve' solves the problem N times with N regularization
                parameters, Lcurve is then fitted on the discrete data points
        scale: scale used to normalize the right side coordinate
                tuple (scale_row, scale_col) when solving jointly, otherwise 
                tuple (scale_row, ) or tuple (scale_col, )
    Returns: 
        theta: the solution vector for the optimal regularization parameter
    '''
    methods = ["initLcurve", "discreteLcurve"]
    if not method in methods: 
        method = "initLcurve"
    if method == "initLcurve": 
        theta, _, _, regu = wlsq_svd(X , y, tol=tol, max_iter=max_iter, plot = plot, scale = scale)
    else: 
        npoints = 100
        eta = np.zeros((npoints,1))
        rho = np.zeros((npoints,1)) 
        thetas = np.zeros((npoints,X.shape[1]))
        reg_param = Lcurve.get_reg_param(X, npoints, truncate = False)
        # solve problem for each regularization param
        for i, h in enumerate(reg_param):
            # the solver is a weighted iterative solver
            sol, norm_res, norm_sol, _ = wlsq_svd(X, y, tol = tol, max_iter = max_iter, h = h, scale = scale)
            thetas[i] = sol.ravel()
            eta[i] = norm_sol
            rho[i] = norm_res
        # find the corner of the Lcurve
        cid = Lcurve.find_corner(eta, rho, plot = plot)
        theta = thetas[cid].reshape(-1,1)
        regu = reg_param[cid]
    return theta, regu

def solve(X, y, tol=1e-2, max_iter=20, method = "initLcurve", plot = False, scale = (1,) ):
    '''
    Solves argmin(theta) || X.theta - y ||^2 + ||theta||^2 
    with ICCV(iteration by correcting characteristic value)
    Initial value determined by Lcurve
    Args:
        X: design matrix
        y: right side term
        max_iter : maximum number of iterations
        tol : tolerance criterion on improvment of RMSE over iterations
        method: the method of finding the regularization h automatically
                'initLcurve' will do a standard tikhonov Lcurve without weights
                to find the h parameter, uses closed form formulas
                'discreteLcurve' solves the problem N times with N regularization
                parameters, Lcurve is then fitted on the discrete data points
        scale: scale used to normalize the right side coordinate
                tuple (scale_row, scale_col) when solving jointly, otherwise 
                tuple (scale_row, ) or tuple (scale_col, )
    Returns: 
        theta: the solution vector for the optimal regularization parameter
    '''
    theta, regu = solve_lc(X, y, tol, max_iter, method, plot, scale )
    theta = theta.reshape((-1,1))
    # get error
    RMSE, weights = compute_rmse(X, y, theta, scale)
    theta_best = theta
    RMSE_best = RMSE
    # ICCV (iteration by correcting characteristic value)
    Id = np.eye(X.shape[1])
    for n_iter in range(1, max_iter+1):
            # iterate
            diag = weights*weights
            theta = np.linalg.inv(X.T @ (diag * X) + Id) @ (X.T @ (diag * y) + theta)
            theta = theta.reshape((-1,1))
            # store previous rmse 
            RMSE_prev = RMSE
            # get rmse and weights
            RMSE, weights = compute_rmse(X, y, theta, scale)
            # store best solution
            if RMSE < RMSE_best: 
                theta_best = theta
                RMSE_best = RMSE
            # check convergence
            if np.abs(RMSE_prev - RMSE) < tol:
                break
    return theta_best, regu

def empty_rpc():
    '''
    Creates an empty rpc instance
    '''
    d = {}
    listkeys = ['LINE_OFF','SAMP_OFF','LAT_OFF','LONG_OFF','HEIGHT_OFF'
           ,'LINE_SCALE','SAMP_SCALE','LAT_SCALE','LONG_SCALE','HEIGHT_SCALE',
           'LINE_NUM_COEFF','LINE_DEN_COEFF','SAMP_NUM_COEFF','SAMP_DEN_COEFF']
    for key in listkeys:
        d[key]= '0'
    return rpc_model.RPCModel(d)

def scaling_params(vect):
    '''
    returns scale, offset based 
    on vect min and max values
    '''
    min_vect = min(vect)
    max_vect = max(vect)
    mid = (max_vect - min_vect)/2 
    return mid, min_vect + mid

def init_rpc(target, input_locs):
    '''
    Initialize an rpc instance on (row, col , lon, lat, h) grids
    by initializing the scale and offset parameters
    '''
    rpc_init = empty_rpc()
    rpc_init.row_scale,rpc_init.row_offset  = scaling_params(target[:,1])
    rpc_init.col_scale,rpc_init.col_offset  = scaling_params(target[:,0])
    rpc_init.lat_scale,rpc_init.lat_offset  = scaling_params(input_locs[:,1])
    rpc_init.lon_scale,rpc_init.lon_offset  = scaling_params(input_locs[:,0])
    rpc_init.alt_scale,rpc_init.alt_offset  = scaling_params(input_locs[:,2])
    return rpc_init
      
def calibrate_rpc( target, input_locs, separate = True, tol=1e-2
                  , max_iter=20, method = "initLcurve", plot = False, 
                  orientation = "projection", get_log = False,
                  init: Optional[rpc_model.RPCModel] = None):
    '''
    fits the coefficients of an RPCModel instance with
    regularized weighted least squares on 3D -> 2D grid correspondence
    Args:
        target: Nx2 (col, row)
        input_locs: Nx3 (lon, lat, alt)
        separate: if True, fit the row and col projection separately
        max_iter : maximum number of iterations
        tol : tolerance criterion on improvment of RMSE over iterations
        method: the method of finding the regularization h automatically
                'initLcurve' will do a standard tikhonov Lcurve without weights
                to find the h parameter, uses closed form formulas
                'discreteLcurve' solves the problem N times with N regularization
                parameters, Lcurve is then fitted on the discrete data points
                using 'discreteLcurve' is discouraged as it is costly
        plot: if True, plot Lcurves
        orientation: "projection" to fit projection rpcs
                     "localization" to fit localization rpcs
                     "projloc" to fit projection and localization rpcs
    Returns: 
        rpc_calib: calibrated RPCModel instance
    '''
    # if not given, initialize an empty rpc instance
    rpc_calib = init or init_rpc(target, input_locs)
    if orientation == "projloc":
        orientlist = ["projection", "localization"]
    else: 
        orientlist = [orientation]
    logger = {}
    for orient in orientlist:
        # get the matrices of the problem
        MR, R , MC, C, A, l = get_mats(rpc_calib, target, input_locs, orientation = orient)     
        # solve jointly or separately
        if separate: 
            thetaR, hr = solve(MR,R,tol,max_iter,method,plot,(rpc_calib.row_scale,))
            thetaC, hc = solve(MC,C,tol,max_iter,method,plot,(rpc_calib.col_scale,))
            logger[orient] = {'hr':hr, 'hc': hc}
        else: 
            theta, h = solve(A,l,tol,max_iter,method,plot, (rpc_calib.row_scale
                                                         ,rpc_calib.col_scale))
            logger[orient] = {'h': h}
            thetaR = theta[:39]
            thetaC = theta[39:]
        coeffs = np.vstack([thetaR[:20], 1, thetaR[20:39],
                            thetaC[:20], 1, thetaC[20:39]]).reshape(-1)
        rpc_calib = update_rpc(rpc_calib, coeffs, orientation = orient)
    if get_log:    
        return rpc_calib, logger
    else:
        return rpc_calib
