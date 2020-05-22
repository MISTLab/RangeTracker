import os, sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'trackerCore'))

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from math import sqrt

savePath = sys.path[0] + "/result_cache/"

from simulationData import readFromFile
from simulationData import sim


def plot_sim(sim, ref_ekf, my_ekf):

    posXRef, posYRef, orientationRef, linVelRef, posX, posY, orientation, linVel = ([] for _ in range(8))

    for idx in range(len(ref_ekf.ekf.recordState)):
        posXRef.append(ref_ekf.ekf.recordState[idx][0])
        posYRef.append(ref_ekf.ekf.recordState[idx][1])
        orientationRef.append(ref_ekf.ekf.recordState[idx][2])
        linVelRef.append(ref_ekf.ekf.recordState[idx][3])

    for idx in range(len(my_ekf.ekf.recordState)):
        posX.append(my_ekf.ekf.recordState[idx][0])
        posY.append(my_ekf.ekf.recordState[idx][1])
        orientation.append(my_ekf.ekf.recordState[idx][2])
        linVel.append(my_ekf.ekf.recordState[idx][3])

    plt.figure(figsize=(7, 7))
    #colors = plt.cm.rainbow(np.linspace(0, 1, len(posX)))
    plt.scatter(sim.x, sim.y, s=2, c='gray', label="Ground Truth")
    plt.scatter(posXRef, posYRef, s=2, c='red', label="EKF without RVE")
    plt.scatter(posX, posY, s=2, c='green', label="Proposed Method")
    plt.scatter(0.,0., marker = '^', s=100, c='b', label="Anchor" )


    gray_patch = mpatches.Patch(color='gray', label="Ground truth")
    red_patch = mpatches.Patch(color='red', label="Without speed estimator")
    green_patch = mpatches.Patch(color='green', label="With speed estimator")
    plt.legend(handles=[gray_patch, red_patch, green_patch])
    plt.xlabel('(m)')
    plt.ylabel('(m)')

    plt.axis('equal')
    plt.savefig(savePath+"sim_result_trajectory.svg")


    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.set_xlabel('Time (steps)')
    ax1.set_ylabel('Linear Velocity(m/s)')
    #ax1.plot(orientation, label="orientation")
    ax1.plot(sim.lVel,c='gray', label=" GT Linear Vel ")
    ax1.plot(linVelRef,c='red', label=" Vanilla EKF Linear Vel. ")
    ax1.plot(linVel, c='green',label=" Proposed Method Linear Vel. ")

    gray_patch = mpatches.Patch(color='gray', label="Ground truth")
    red_patch = mpatches.Patch(color='red', label="Without speed estimator")
    green_patch = mpatches.Patch(color='green', label="With speed estimator")
    ax1.legend(loc='upper left', handles=[gray_patch, red_patch, green_patch])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #ax2.plot(uwbInput, color='c', label=" Range Measurement ")
    ax2.plot(my_ekf.speedEstimator.filtedRange, color='goldenrod', label=" Filtered range ")
    ax2.set_ylabel('Simulated UWB range(m)')
    glodenrod_patch = mpatches.Patch(color='goldenrod', label="Range")
    ax2.legend(loc='upper right', handles=[glodenrod_patch])

    fig.savefig(savePath+"sim_result_info.svg")



def plot_sim_error(sim, ref_ekf, my_ekf):

    posXRef, posYRef, orientationRef, linVelRef, posX, posY, orientation, linVel = ([] for _ in range(8))

    error  = []
    errorRef = []


    for idx in range(len(ref_ekf.ekf.recordState)):
        posXRef.append(ref_ekf.ekf.recordState[idx][0])
        posYRef.append(ref_ekf.ekf.recordState[idx][1])
        orientationRef.append(ref_ekf.ekf.recordState[idx][2])
        linVelRef.append(ref_ekf.ekf.recordState[idx][3])
        errorRef.append(np.linalg.norm([posXRef[idx] - sim.x[idx],posYRef[idx]-sim.y[idx]]))

    for idx in range(len(my_ekf.ekf.recordState)):
        posX.append(my_ekf.ekf.recordState[idx][0])
        posY.append(my_ekf.ekf.recordState[idx][1])
        orientation.append(my_ekf.ekf.recordState[idx][2])
        linVel.append(my_ekf.ekf.recordState[idx][3])
        error.append(np.linalg.norm([posX[idx] - sim.x[idx],posY[idx]-sim.y[idx]]))

    plt.figure(figsize=(10 , 5))
    plt.plot(errorRef,color='red',  label = ' Vanilla EKF')
    plt.plot(error, color='green', label = 'Proposed Method')

    red_patch = mpatches.Patch(color='red', label="Without speed estimator")
    green_patch = mpatches.Patch(color='green', label="With speed estimator")
    plt.legend(loc='upper left', handles=[red_patch, green_patch])
    plt.xlabel('Time (steps)')
    plt.ylabel('RMSE(m)')

    print("RMSE With Speed Estimator", np.mean(error), "; Without", np.mean(errorRef))
    plt.savefig(savePath+"sim_RMS.svg")



def vel_from_dis( l_0, l_1, l_2, t0, t1, t2):
    t_1 = t1 - t0
    t_2 = t2 - t1
    d = abs(l_2 * l_2 - l_1 * l_1 - (l_1 * l_1 - l_0 * l_0) * t_2 / t_1)
    tl = t_1 * t_2 + t_2 * t_2
    return sqrt(d / tl)


def brute_vel_estimate(range_measurement, bVel, interval=500):
    interval = 50
    dt = 0.005
    for i in range( 2*interval, len(range_measurement)):
        t0 = i * dt
        t1 = (i + interval) * dt
        t2 = (i + interval * 2) * dt
        bVel.append(
            vel_from_dis(range_measurement[i-2*interval], range_measurement[i - interval], range_measurement[i], t0, t1, t2))


if __name__ == '__main__':
    simData = readFromFile()