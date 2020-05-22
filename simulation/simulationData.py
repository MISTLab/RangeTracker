
# Perfect simulation data to test
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'trackerCore'))

import numpy as np
from math import pi, sin, cos, atan2
import random
from matplotlib import pyplot as plt
from utility import normalizeAngle 
import pickle

curPath = sys.path[0]

def saveToFile(name):
    with open(curPath+'/result_cache/diffWheelRobot.pkl', 'wb') as output:
        pickle.dump(name, output, pickle.HIGHEST_PROTOCOL)
    output.close()

def readFromFile():
    with open(curPath+'/result_cache/diffWheelRobot.pkl', 'rb') as input:
       return pickle.load(input)


class sim():
    def __init__(self):
        self.x = [10., ]
        self.y = [0., ]
        self.dt = 0.005
        self.lVel = [0.0]
        self.aVel = [0.0, ]
        self.vX = [-0.3]
        self.vY = [0.]
        self.refBearing = [0.0, ]
        self.bearing = [0.0, ]
        self.uwb = [np.linalg.norm([self.x[0], self.y[0]])]
        self.uwbNoisy = [np.linalg.norm([self.x[0], self.y[0]])]
        self.yaw = [normalizeAngle(pi / 2 + self.aVel[0] * self.dt), ]
        self.yawNoisy = [normalizeAngle(pi / 2 + self.aVel[0] * self.dt),]
        self.timestamp = [0.]
        self.lAcc = [0.]
        self.dataSize = 10000
        self.numDynamics = 5
        self.targetLV = [0.0, 2.4, 1.7, 1.2, 3.1, 2.0, 2.7]
        self.targetAV = [0.0, 1.5, -1.9, 2., 2.9, 1., 0.6]
        self.dynamicTime = [0, 1000, 2000, 3000, 4000, 5000, 6000]
        self.adaptionPeriod = 50
        self.period = 0
        self.dynamicStart = 0
        self.doDynamics = False
        self.notSetStep = True
        self.notSetPeriod = True

    def generate_sim_root(self):
        self.targetLV = [0.]
        self.targetAV = [0.]
        self.dynamicTime = [0]

        for _ in range(self.numDynamics):
            lv = random.randint(1, 100)/10
            av = random.randint(1, 100)/10
            evenDistributedInterval = int(self.dataSize / self.numDynamics)
            interval = random.randint(evenDistributedInterval/2, evenDistributedInterval)
            self.targetLV.append(lv)
            self.targetAV.append(av)
            self.dynamicTime.append(int(interval+self.dynamicTime[-1]))

        pass

    def check_step_dymics(self, step):
        if self.dynamicTime[self.period]+self.adaptionPeriod > step > self.dynamicTime[self.period]:
            if self.notSetStep:
                self.reset_step(step)
            self.notSetPeriod = True
            return True
        else:
            if self.notSetPeriod:
                self.reset_period()
            self.notSetStep = True
            return False

    def reset_period(self):

        self.period = self.period + 1 if self.period < self.numDynamics else self.numDynamics
        self.notSetPeriod = False

    def reset_step(self,step):
        self.dynamicStart = step
        self.notSetStep = False

    def generate_sim(self):
        for step in range(self.dataSize):
            self.timestamp.append(step*self.dt)
            if self.check_step_dymics(step):
                interval = step - self.dynamicStart
                x = pi*interval/self.adaptionPeriod
                lv = self.targetLV[self.period -1] + sin(x/2)* (self.targetLV[self.period]-self.targetLV[self.period-1])
                av = sin(x) * self.targetAV[self.period]
            else:
                lv = self.lVel[step]
                av = 0.

            self.lAcc.append((lv-self.lVel[-1])/self.dt)
            self.lVel.append(lv)
            self.aVel.append(av)

            yaw_now = normalizeAngle(self.yaw[step] + av * self.dt)
            yaw_noisy = yaw_now + np.random.normal(0, 0.01)
            self.yaw.append(yaw_now)
            self.yawNoisy.append(yaw_noisy)
            self.vX.append(lv * cos(yaw_now))
            self.vY.append(lv * sin(yaw_now))
            del_displacement = lv * self.dt

            x_now = self.x[step] + del_displacement * cos(yaw_now)
            y_now = self.y[step] + del_displacement * sin(yaw_now)
            self.x.append(x_now)
            self.y.append(y_now)

            ref_bearing = atan2(y_now, x_now)
            self.refBearing.append(ref_bearing)
            bearing = normalizeAngle(pi - yaw_now + ref_bearing)
            self.bearing.append(bearing)

            range_meas = np.linalg.norm([self.x[step], self.y[step]])
            self.uwb.append(range_meas)
            range_noisy = range_meas + np.random.normal(0, 0.005)
            self.uwbNoisy.append(range_noisy)


    def plot_sim(self):
        plt.figure(figsize=(20, 10))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.x)))
        plt.scatter(self.x, self.y, c=colors, s=2)

        for step in self.dynamicTime:
            plt.annotate(step, (self.x[step], self.y[step]))
        plt.axis('equal')
        plt.scatter(0., 0., marker='^', color='r', s=20)
        plt.savefig(curPath+"/result_cache/groundTruthTrajectory.svg")

        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('Orientation/ aVel/ lVel')
        ax1.plot(self.yaw, label="Yaw")
        ax1.plot(self.aVel, label=" Angular Vel ")
        ax1.plot(self.lVel, label=" Linear Vel ")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(self.uwb, color='c', label=" simUwb ")
        ax2.set_ylabel('UWB range')
        fig.legend()
        fig.savefig(curPath+"/result_cache/groundTruthInfo.svg")

if __name__ == '__main__':
    #simData = sim()
    #simData.generate_sim_root()
    # simData.generate_sim()
    # simData.plot_sim()
    # saveToFile(simData)
    simData = readFromFile()
