import numpy as np

def sq_err(w, u, rotation, cor):
	"""
	cor:	the correlation matrix (of the axis aligned) Gaussian distribution. So
			cor is diagonal. We assume that each entry of cor is between 0.1 and 1
	rotation: the rotation matrix applied to vectors drawn from N(0, cor)
	w, u : vectors in the transformed (rotated space)
	
	NOTE: All of cor, rotation, w, u should be declared as np.mat. cor and
	rotation are assumed to be nxn and w, u assumed to be nx1.

	Outputs: E_{y ~ N(0, rotation*cor*rotation^{-1}}[(w*y - u*y)^2]
	"""
	return (((w - u).T)*rotation*cor*(rotation.I)*(w - u))[0, 0]
