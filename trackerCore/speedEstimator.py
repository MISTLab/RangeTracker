from collections import deque
import numpy as np
from KalmanFilter import KalmanFilter
import math



class speedEstimator():
    def __init__(self):
        self.validSpeedUpdated = False
        self.lastPoint = 0.
        self.distanceThreshold = 0.05
        self.speedUpdateTime = []
        self.speedWindowSize = 10
        self.keyMeasPairs = deque(maxlen=80)
        self.speedWindow = []
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
        self.speedWindowSize = 5+ 0.1*range

        if self.range_key_pairs_maintaince(fdragne, time) and len(self.keyMeasPairs) >= 2*interval:
            tempresult = self.vel_from_dis(self.keyMeasPairs[-2*interval][0], self.keyMeasPairs[-interval][0],
                                           self.keyMeasPairs[-1][0], self.keyMeasPairs[-2*interval][1],
                                           self.keyMeasPairs[-interval][1], self.keyMeasPairs[-1][1])
            if tempresult:
                self.speedWindow.append(tempresult)
                if len(self.speedWindow)>(self.speedWindowSize -1):
                    self.curSpeed = np.median(self.speedWindow)  # Estimation of this linear motion speed
                    self.speedRecord.append(self.curSpeed)
                    self.speedUpdateTime.append(time)
                    self.validSpeedUpdated = True
                    while(len(self.speedWindow)>(self.speedWindowSize -1)):
                        self.speedWindow.pop(0)

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
