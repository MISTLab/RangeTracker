import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'trackerCore'))

from tracker import tracker
from simulationData import sim, readFromFile, saveToFile
from plot import plot_sim, plot_sim_error

if __name__ == '__main__':

    ##Generate new simulation data

    # simData = sim()
    # simData.generate_sim_root()
    # simData.generate_sim()
    # simData.plot_sim()
    # saveToFile(simData)

    simData = readFromFile()

    myTrackerExp = tracker()
    refMyTracker = tracker()

    simulation = True
    myTrackerExp.setup_mode(simulation)
    speedEstimatorSwitch = False
    refMyTracker.setup_mode(simulation,speedEstimatorSwitch)

    covS_X = 0.01  # covariance of state
    covS_Y = 0.01
    covS_Ori = 0.01
    covS_LVel = 0.01

    covM_Range = 0.1  # Covariance of range measurement
    covM_Ori = 0.1

    refMyTracker.ekf.set_covs(covS_X, covS_Y, covS_Ori, covS_LVel, covM_Range, covM_Ori)
    myTrackerExp.ekf.set_covs(covS_X, covS_Y, covS_Ori, covS_LVel, covM_Range, covM_Ori)

    initialState = [10., 0., 0., 0.0]  
    refMyTracker.ekf.set_initial_state(initialState)
    myTrackerExp.ekf.set_initial_state(initialState)


    uwbInput = simData.uwbNoisy
    yawInput = simData.yawNoisy
    timeInput = simData.timestamp

    print("Start the Tracker")
    for step in range(len(uwbInput)):
        measurement = [uwbInput[step], yawInput[step], timeInput[step]]
        refMyTracker.step(measurement)
        myTrackerExp.step(measurement)

    plot_sim(simData,refMyTracker,myTrackerExp)
    plot_sim_error(simData,refMyTracker,myTrackerExp)