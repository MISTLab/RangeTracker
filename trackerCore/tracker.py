from KalmanFilter import KalmanFilter
from ExtendedKalmanFilter import ExtendedKalmanFilter
import numpy as np
import math
from math import sin, cos, pi
from collections import deque
from utility import uwbPassOutlierDetector, normalizeAngle
from speedEstimator import speedEstimator



class tracker():
    def __init__(self,lineMovingThreshold=0.1):
        self.speedEstimator = speedEstimator()
        self.ekf = customizedEKF()
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



    def setup_mode(self, simulation, speedEstimatorSwitch = True):
        self.isSimulation = simulation
        self.withSpeedEstimator = speedEstimatorSwitch
        if self.isSimulation:
            self.speedIinterval = 10
        else:
            self.speedIinterval = 20


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

    def update_sim_measurement_range(self, rangeMeas, time):
        self.newMeasRange = rangeMeas
        self.rangeMeasUpdated = True
        self.measUpdateTime = time

    def update_heading_measurement(self, headingMeas, time):
        self.newMeasHeading = headingMeas
        self.measUpdateTime = time
        self.headingMeasUpdated = True


    def linear_motion_check(self):
        if abs(self.newMeasHeading-self.curHeading) < self.lineMovingThreshold:
            return True
        else:
            self.curHeading = self.newMeasHeading
            self.speedEstimator.keyMeasPairs.clear()
            return False

    def real_step(self,measurement):
        rangemeas = measurement[0]
        headmeas = measurement[1]
        timeStamp = measurement[2]
        acc = measurement[3]
        if self.get_valid_measurement_range(rangemeas, timeStamp):
            self.speedEstimator.estimate_speed(measurement[0], timeStamp, self.speedIinterval)
        self.update_heading_measurement(headmeas,timeStamp)
        if self.check_static(acc):
            self.ekf.x[3] = 0
            self.speedEstimator.keyMeasPairs = []
        else:
            self.ekf.ekfStep([self.newMeasRange, self.newMeasHeading])
            if self.withSpeedEstimator:
                if self.speedEstimator.validSpeedUpdated:
                    estimatedVel = self.speedEstimator.get_vel()
                    self.ekf.x[3] = 0.5*self.ekf.x[3] + 0.5*estimatedVel
            self.ekf.records()

    def sim_step(self, measurement):
        rangemeas = measurement[0]
        headmeas = measurement[1]
        timeStamp = measurement[2]
        self.update_sim_measurement_range(rangemeas,timeStamp)
        self.speedEstimator.estimate_speed(rangemeas, timeStamp, self.speedIinterval)
        self.update_heading_measurement(headmeas,timeStamp)
        self.ekf.ekfStep([self.newMeasRange, self.newMeasHeading])
        if self.withSpeedEstimator and self.linear_motion_check() and self.speedEstimator.validSpeedUpdated:
            estimatedVel = self.speedEstimator.get_vel()
            self.ekf.x[3] = 0.5*self.ekf.x[3] + 0.5*estimatedVel
        self.ekf.records()


    def step(self, measurement):
        if self.isSimulation:
            self.sim_step(measurement)
        else:
            self.real_step(measurement)


class customizedEKF(ExtendedKalmanFilter):
    
    def __init__(self, dim_x=4, dim_z=2):
        super(customizedEKF, self).__init__(dim_x, dim_z)
        self.dt = 0.005
        self.recordState = []
        self.recordResidual = []
        self.recordP = []

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

    def predict_x(self, state):
        x = self.x[0]
        y = self.x[1]
        o = self.x[2]
        v = self.x[3]

        self.x[0] = x + v * cos(o) * self.dt
        self.x[1] = y + v * sin(o) * self.dt
        self.x[2] = o
        self.x[3] = v

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
        self.calF()
        self.predict()
        self.x[2] = normalizeAngle(self.x[2])
        self.update(measurement, self.H_Jac, self.H_state, residual=self.residualWithAng)

    def records(self):
        self.recordState.append(self.x)
        self.recordP.append(self.P)
        self.recordResidual.append(self.y)
