import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'trackerCore'))

from tracker import tracker
from simulationData import sim, readFromFile, saveToFile
from plot import plot_sim, plot_sim_error
from parameters import *

if __name__ == '__main__':

    #Using last time data or generate new simulation data
    if UsingLastTimeData:
        simData = readFromFile()
    else:
        simData = sim()
        simData.generate_sim_root()
        simData.generate_sim()
        saveToFile(simData)

    # Create two tracker for compare
    myTrackerExp = tracker()
    refMyTracker = tracker()

    # Set tracker mode to simulation, disable the speedEstimator of the reference tracker
    myTrackerExp.setup_mode(IsSimulation)
    speedEstimatorSwitch = False
    refMyTracker.setup_mode(IsSimulation,speedEstimatorSwitch)

    # Set covariance for the EKF
    refMyTracker.ekf.set_covs(covS_X, covS_Y, covS_Ori, covS_LVel, covM_Range, covM_Ori)
    myTrackerExp.ekf.set_covs(covS_X, covS_Y, covS_Ori, covS_LVel, covM_Range, covM_Ori)

    # Set initial state for the EKF
    refMyTracker.ekf.set_initial_state(initialState)
    myTrackerExp.ekf.set_initial_state(initialState)

    # Choose measurement input
    uwbInput = simData.uwbNoisy
    yawInput = simData.yawNoisy
    timeInput = simData.timestamp

    print("Start the Tracker")
    for step in range(len(uwbInput)):
        measurement = [uwbInput[step], yawInput[step], timeInput[step]]
        refMyTracker.step(measurement)
        myTrackerExp.step(measurement)

    # Plot the result to result_cache folder
    plot_sim(simData,refMyTracker,myTrackerExp)
    plot_sim_error(simData,refMyTracker,myTrackerExp)