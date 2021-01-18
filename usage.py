#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:02:21 2021

@author: rakiki
"""
from RpcFit import rpc_fit
import numpy as np

locs_train = np.load('./data/s1_train3d.npy')
target_train = np.load('./data/s1_train2d.npy')

locs_test = np.load('./data/s1_test3d.npy')
target_test = np.load('./data/s1_test2d.npy')


# fit on training set
rpc_calib, log = rpc_fit.calibrate_rpc(target_train, locs_train, separate=False, tol=1e-10
                                      , max_iter=20, method='initLcurve'
                                      , plot=True, orientation = 'projloc', get_log=True )

# evaluate on training set
rmse_err, mae, planimetry = rpc_fit.evaluate(rpc_calib, locs_train, target_train)
print('Training set :   Mean X-RMSE {:e}     Mean Y-RMSE {:e}'.format(*rmse_err))

# evaluate on the test set
rmse_err, mae, planimetry = rpc_fit.evaluate(rpc_calib, locs_test, target_test)
print('Test set :   Mean X-RMSE {:e}     Mean Y-RMSE {:e}'.format(*rmse_err))
