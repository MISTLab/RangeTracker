#! /usr/bin/env python
import re
from transformations import euler_from_quaternion
import numpy as np
import math
from numpy import linalg as la


def readData(sourceFileName, uwbTime, uwb, imuTime, yaw, acc, gyro, filtedOri):
	expEuler = []
	f = open(sourceFileName,'r')
	print(sourceFileName)
	numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
	rx = re.compile(numeric_const_pattern, re.VERBOSE)
	with open(sourceFileName, 'r') as f:
		for line in f:
			dis=rx.findall(line)
			#print(dis)
			if line[0]=="u":
				temp = float(dis[0]) + float(dis[1])*10**-9 # time tamp of imu measurement
				uwbTime.append(temp)
				uwb.append(float(dis[2]) / 1000)
				#print("add uwb", uwb[-1])
				pass
			elif line[0] == "i":
				temp = float(dis[0]) + float(dis[1])*10**-9 # time tamp of imu measurement
				imuTime.append(temp)

				quater = [float(dis[3]),float(dis[4]),float(dis[5]), float(dis[2])] # x-y-z-w
				expEuler = euler_from_quaternion(quater)
				yaw.append(expEuler[2])

				temp = []
				temp = [float(dis[6]),float(dis[7]),float(dis[8])]
				acc.append(temp)

				temp = []
				temp = [float(dis[9]),float(dis[10]),float(dis[11])]
				gyro.append(temp)
				pass

	#gtfile = pd.read_csv(optitrackFileName, header=0, skiprows=6, engine='python')
	print("data size -range ", len(uwb), " angle: ", len(yaw))
	pass



def readM100Data(sourceFileName, uwbTime, uwb, imuTime, yaw, acc, gyro, velTime, vel, pos):

# cloct data for ekf
	expEuler = []
	file = open(sourceFileName, 'r')
	print(sourceFileName)
	numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
	rx = re.compile(numeric_const_pattern, re.VERBOSE)
	with open(sourceFileName, 'r') as file:
		for line in file:
			dis=rx.findall(line)
			#print(dis)
			if line[0]=="u":
				temp = float(dis[0]) + float(dis[1])*10**-9 # time tamp of imu measurement
				uwbTime.append(temp)
				uwb.append(float(dis[2]) / 1000)
				#print("add uwb", uwb[-1])
				pass
			elif line[0] == "i":
				temp = float(dis[0]) + float(dis[1])*10**-9 # time tamp of imu measurement
				imuTime.append(temp)

				quater = [float(dis[3]),float(dis[4]),float(dis[5]), float(dis[2])] # x-y-z-w
				expEuler = euler_from_quaternion(quater)
				yaw.append(expEuler[2])

				temp = []
				temp = [float(dis[6]),float(dis[7]),float(dis[8])]
				acc.append(temp)

				temp = []
				temp = [float(dis[9]),float(dis[10]),float(dis[11])]
				gyro.append(temp)
				pass

			elif line[0] == "v":
				temp = float(dis[0]) + float(dis[1])*10**-9 # time tamp of imu measurement
				velTime.append(temp)
				temp = [float(dis[2]),float(dis[3]),float(dis[4])]
				vel.append(temp)
				pass

			elif line[0] == "p":
				temp = [float(dis[2]), float(dis[3]), float(dis[4])]
				pos.append(temp)
				pass

	#gtfile = pd.read_csv(optitrackFileName, header=0, skiprows=6, engine='python')
	print("data size -range ", len(uwb), " angle: ", len(yaw))
	pass


def normalizeAngle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


# data_collection is a deque, with fixed size.
# 0.4 is the threshold
# 10 is incase the outlier has already be added to data_collection.

def uwbPassOutlierDetector(data_collection, newVal):
	data_collection.append(newVal)
	if np.std(data_collection) < 0.2:
		return True
	elif len(data_collection) < 10:
		data_collection.clear()
		return False
	else:
		data_collection.pop()
		return False



def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
