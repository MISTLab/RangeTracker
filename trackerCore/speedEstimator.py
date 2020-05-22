from KalmanFilter import KalmanFilter
from ExtendedKalmanFilter import ExtendedKalmanFilter
import numpy as np
import math
from math import sin, cos, pi
from collections import deque
from utility import uwbPassOutlierDetector, normalizeAngle


class tracker():
    def __init__(self,lineMovingThreshold=0.1):
        self.speedEstimator = speedEstimator()
        self.ekf = myalgEkf()
        self.accData = deque(maxlen=100)
        self.staticThreshold = 0.01
        self.isSimulation = False

        self.lineMovingThreshold = lineMovingThreshold
        self.curHeading = 0.
        self.measUpdateTime = 0.
        self.newMeasHeading = 0.
        self.newMeasRange = 0.
        self.histRange = deque(maxlen=20)

        self.withSpeedEstimator = True



    def setupMode(self, simulation, speedChoice = True):
        self.isSimulation = simulation
        self.withSpeedEstimator = speedChoice
        if self.isSimulation:
            self.speedIinterval = 10
        else:
            self.speedIinterval = 20

        pass


    def check_static(self, acc):
        self.accData.append(acc)
        if np.std(self.accData) > self.staticThreshold:
            return False
        else:
            return True



    def get_valid_measurement_range(self, rangeMeas, time):
        if uwbPassOutlierDetector(self.histRange, rangeMeas):
            calibUWB = 1.11218892 * rangeMeas - 0.03747436  # TUM basketball calibration result
            self.newMeasRange = calibUWB
            self.rangeMeasUpdated = True
            self.measUpdateTime = time
            return True
        else:
            return False

    def get_sim_measurement_range(self, rangeMeas, time):
        self.newMeasRange = rangeMeas
        self.rangeMeasUpdated = True
        self.measUpdateTime = time

    def get_heading_measurement(self, headingMeas, time):
        self.newMeasHeading = headingMeas
        self.measUpdateTime = time
        self.headingMeasUpdated = True
        pass


    def linear_motion_check(self):
        if abs(self.newMeasHeading-self.curHeading) < self.lineMovingThreshold:
            return True
        else:
            self.curHeading = self.newMeasHeading
            self.speedEstimator.keyMeasPairs.clear()
            return False

    def realStep(self,measurement):
        rangemeas = measurement[0]
        headmeas = measurement[1]
        timeStamp = measurement[2]
        acc = measurement[3]
        if self.get_valid_measurement_range(rangemeas, timeStamp):
            self.speedEstimator.estimate_speed(measurement[0], timeStamp, self.speedIinterval)
        self.get_heading_measurement(headmeas,timeStamp)
        if self.check_static(acc): # reset if static
            self.ekf.x[3] = 0
            self.speedEstimator.keyMeasPairs = []
        else:
            self.ekf.ekfStep([self.newMeasRange, self.newMeasHeading])
            if self.withSpeedEstimator:
                #if self.linear_motion_check() and self.speedEstimator.validSpeedUpdated:
                #if self.linear_motion_check():
                if self.speedEstimator.validSpeedUpdated:
                    select_vel = self.speedEstimator.get_vel()
                    self.ekf.x[3] = 0.5*self.ekf.x[3] + 0.5*select_vel
                    #self.ekf.x[3] = select_vel
                #self.speedEstimator.speedRecord.append(self.ekf.x[3])
            self.ekf.records()

    def simStep(self, measurement):
        rangemeas = measurement[0]
        headmeas = measurement[1]
        timeStamp = measurement[2]
        self.get_sim_measurement_range(rangemeas,timeStamp)
        self.speedEstimator.estimate_speed(rangemeas, timeStamp, self.speedIinterval)
        self.get_heading_measurement(headmeas,timeStamp)
        self.ekf.ekfStep([self.newMeasRange, self.newMeasHeading])
        #if self.withSpeedEstimator:
        if self.withSpeedEstimator and self.linear_motion_check() and self.speedEstimator.validSpeedUpdated:
            select_vel = self.speedEstimator.get_vel()
            self.ekf.x[3] = 0.5*self.ekf.x[3] + 0.5*select_vel
            #self.ekf.x[3] = select_vel
            #self.speedEstimator.speedRecord.append(self.ekf.x[3])
        self.ekf.records()

        pass


    def step(self, measurement):
        if self.isSimulation:
            self.simStep(measurement)
        else:
            self.realStep(measurement)





class speedEstimator(object):

    # This is a parallel velocity estimator from UWB range measurement

    def __init__(self):
        super(speedEstimator, self).__init__()
        self.validSpeedUpdated = False
        self.lastPoint = 0.
        self.rangeSlideWindow = deque(maxlen=10)
        self.distanceThreshold = 0.05

        self.speedUpdateTime = []

        self.keyMeasPairs = deque(maxlen=80)
        self.speedWindow = deque(maxlen=10)
        self.filtedRange = []
        self.curSpeed = 0.0
        self.speedRecord = []

        self.rangeKF = KalmanFilter(dim_x=1, dim_z=1)
        self.rangeKF.x = np.array([0.])
        self.rangeKF.F = np.array([[1.]])
        self.rangeKF.H = np.array([[1.]])
        self.rangeKF.P *= 100.
        self.rangeKF.R = 0.1
        self.rangeKF.Q = 0.001
        self.rangeKF.initialized = False

    def vel_from_dis(self, l_0, l_1, l_2, t0, t1, t2):
        t_1 = t1 - t0
        t_2 = t2 - t1
        if (l_2 - l_1) * (l_1 - l_0) > 0:
            d = abs(l_2 * l_2 - l_1 * l_1 - (l_1 * l_1 - l_0 * l_0) * t_2 / t_1)
            tl = t_1 * t_2 + t_2 * t_2
            return math.sqrt(d / tl)
        else:
            return False


    def range_key_pairs_maintaince(self, range, time):
        if abs(range-self.lastPoint) > self.distanceThreshold:
            self.lastPoint = range
            self.keyMeasPairs.append([range, time])
            return True
        else:
            return False


    def estimate_speed(self, range, time, interval):
        fdragne = self.filter_range(range)

        if self.range_key_pairs_maintaince(fdragne, time) and len(self.keyMeasPairs) >= 2*interval:
            tempresult = self.vel_from_dis(self.keyMeasPairs[-2*interval][0], self.keyMeasPairs[-interval][0],
                                           self.keyMeasPairs[-1][0], self.keyMeasPairs[-2*interval][1],
                                           self.keyMeasPairs[-interval][1], self.keyMeasPairs[-1][1])
            if tempresult:
                self.speedWindow.append(tempresult)
                self.curSpeed = np.median(self.speedWindow)  # Estimation of this linear motion speed
                self.speedRecord.append(self.curSpeed)
                self.speedUpdateTime.append(time)
                self.validSpeedUpdated = True
        else:
            self.validSpeedUpdated = False





    def filter_range(self, range):
        if self.rangeKF.initialized == False:
            self.rangeKF.x = np.array([range])
            self.filtedRange.append(range)
            self.rangeKF.initialized = True
        else:
            self.rangeKF.predict()
            self.rangeKF.update(range)
            self.filtedRange.append(self.rangeKF.x)
        return self.filtedRange[-1]


    def get_vel(self):
        return self.curSpeed

    def ontimefilter(self, source, result):
        tempEkf = KalmanFilter(dim_x=1, dim_z=1)
        tempEkf.x = np.array([source[0]])
        tempEkf.F = np.array([[1.]])
        tempEkf.H = np.array([[1.]])
        tempEkf.P *= 100.
        tempEkf.R = 0.1
        tempEkf.Q = 0.00001
        for i in range(len(source)):
            tempEkf.predict()
            tempEkf.update(source[i])
            result.append(tempEkf.x)

class myalgEkf(ExtendedKalmanFilter):
    
    def __init__(self, dim_x=4, dim_z=2):
        super(myalgEkf, self).__init__(dim_x=4, dim_z=2)

        covS_X = 0.1  # covariance of range state
        covS_Y = 0.1
        covS_Ori = 0.1
        covS_LVel = 0.1

        covM_Range = 0.1  # Covariance of range measurement
        covM_Ori = 0.1
        self.dt = 0.005
        self.Q = np.array([[covS_X, 0., 0.],  # process uncertainty
                           [0., covS_Y, 0.],
                           [0., 0., covS_Ori]
                           ])

        self.R = np.array([[covM_Range, 0.],
                           [0., covM_Ori],
                           ])

        self.recordState = []
        self.recordResidual = []
        self.recordP = []
        pass

    def set_covs(self, covS_X, covS_Y, covS_Ori, covS_LVel, covM_Range, covM_Ori):
        self.Q = np.array([[covS_X, 0., 0., 0.],
                               [0., covS_Y, 0., 0.],
                               [0., 0., covS_Ori, 0.],
                               [0., 0., 0., covS_LVel]])

        self.R = np.array([[covM_Range, 0.],
                               [0, covM_Ori],
                               ])

    def set_initial_state(self, initialState):
        self.x = initialState
        pass

    def predict_x(self, state):
        x = self.x[0]
        y = self.x[1]
        o = self.x[2]
        v = self.x[3]

        self.x[0] = x + v * cos(o) * self.dt
        self.x[1] = y + v * sin(o) * self.dt
        self.x[2] = o
        self.x[3] = v
        pass

    def calF(self):
        x = self.x[0]
        y = self.x[1]
        o = self.x[2]
        v = self.x[3]

        dx_dx = 1.
        dx_dv = cos(o) * self.dt
        dx_do = -sin(o) * v * self.dt

        dy_dy = 1.
        dy_dv = sin(o) * self.dt
        dy_do = cos(o) * v * self.dt

        self.F = np.array([[dx_dx, 0., dx_do, dx_dv, ],
                           [0., dy_dy, dy_do, dy_dv],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])

        pass

    def residualWithAng(self, zmeas, zpre):
        pi = math.pi
        resVal = np.subtract(zmeas, zpre)
        resVal[1] = normalizeAngle(resVal[1])
        return resVal

    def H_Jac(self, s):
        xnorm = np.linalg.norm([self.x[0], self.x[1]])
        dr_dx = self.x[0] / xnorm
        dr_dy = self.x[1] / xnorm
        Hjac = np.array([[dr_dx, dr_dy, 0, 0],
                         [0., 0., 1., 0.],
                         ])
        return Hjac

    def H_state(self, s):
        xnorm = np.linalg.norm([self.x[0], self.x[1]])
        h_x = np.array([xnorm, self.x[2]])
        return h_x

    def ekfStep(self, measurement):
        # Measurement format: measurements_0 =  np.array([float(uwbR[step]),float(yawmeas[step]), measLVel[step]])
        self.calF()
        self.predict()
        self.x[2] = normalizeAngle(self.x[2])
        self.update(measurement, self.H_Jac, self.H_state, residual=self.residualWithAng)
        pass

    def records(self):
        self.recordState.append(self.x)
        self.recordP.append(self.P)
        self.recordResidual.append(self.y)
