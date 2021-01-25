# RpcFit
The python module corresponding to the igarss2021 paper submission 
<br> Robust Rational Polynomial Camera Modelling for SAR and Pushbroom Imaging
<br>install requirements : pip install -r requirements.txt
<br>Then test installation with: python usage.py

<br>The package contains: 
<br>'gridata.py': Has the necessary functions to construct the 3D+2D point grid correspondence (CoNtrol Point & ChecK Point). Needs a projection function or a localization function from a physical sensor model. A physical sensor model for the Sentinel-1 satellite may be made available in the future. otherwise, consider using snappy for Sentinel-1
<br>'Lcurve.py': Has the necessary function to do the Lcurve criterion with the standard tikhonov, or with a set of discrete points by using the spline curve interpolation
<br>'rpc_fit': Has the necessary functions to fit the rpc on the constructed grids. use calibrate_rpc. other functions are internal helper funcs

<br>The data folder contains datasets for the test in usage.py: 
<br>2d refers to image coordinates 
<br>3d refers to lon, lat, height coordinates
<br>train refers to the training set
<br>test refers to the test set



