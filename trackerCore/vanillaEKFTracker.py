#! /usr/bin/env python

import rospy
import tf
import numpy as np
from filterlib import ExtendedKalmanFilter
import math
from filterDerivFuncs import H_Jac_31_eh01  , H_meas_31_eh01
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3Stamped

"""
EKF model

State: (x,y,v_x, v_y)

State definition.

State: (x,y,dot_x,dot_y)
Measurement: ( range, yaw)

Transfer functions:

x_{t+1} = x_t + dot_x*t
y_{t+1} = y_t + dot_y*t
dot_x_{t+1} = dot_x_{t}
dot_y_{t+1} = dot_y_{t}


Measurement functions:

range_t = sqrt{(x_t^2 + y_t^2)}
yaw_t = arctan(dfr_c {dot_y_t} {dot_x_t})


"""


class Tracker(object):
	"""docstring for enhancedEKF"""
	def __init__(self):
		super(Tracker, self).__init__()


		self.anchorPos = [0.,0.]

		self.rangeMeasCov = 0.1
		self.yawMeasCov = 0.02


		# Initialize the filter
		self.filter = ExtendedKalmanFilter(dim_x=3, dim_z=1)

		self.suberRagne = rospy.Subscriber("/uwb/rawrange",Float64, self.rangeUpdate)
		self.suberImu = rospy.Subscriber("/imu/data_raw",Imu, self.imuUpdate)
		self.suberYaw = rospy.Subscriber("/imu/rpy/filtered",Vector3Stamped,self.yawUpdate)

		self.pub_vel = rospy.Publisher('/vanilarEKF/linearVel', Float64, queue_size=10)

		self.ekfInitialized = False
		self.isImuUpdated = False
		self.isRangeUpdated = False
		self.isYawUpdated = False

		self.state = []
		self.dt = 0.05
		self.lastTime = 0.0
		self.posCov = 0.1
		self.velCov = 0.1

		self.yawMeas = 0.0
		self.rangeMeas = 0.0

		self.vtest = 0.0
		self.vtestLT = 0.0
		self.lastAV = 0
		self.lastImu = Imu()


	def efkInitialization(self):

		while not self.ekfInitialized and not rospy.is_shutdown():
			if self.isYawUpdated and self.isRangeUpdated:		

				initX = self.rangeMeas*math.sin(self.yawMeas)
				initY = - self.rangeMeas*math.cos(self.yawMeas)
				initVel = 0.0



				self.filter.x = np.array([  [initX ],
											[initY ],
											[initVel]])

				self.filter.Q = np.diag([ self.posCov, self.posCov, self.velCov])  # Process noise

				self.filter.R = np.diag([ self.rangeMeasCov] ) # Measurement noise 

				self.filter.P *= 0.1

				self.ekfInitialized = True

				self.lastTime = rospy.Time.now()

				print(" time now ", self.lastTime )


		pass

	def ekfLoop(self):


		while not rospy.is_shutdown():
			if self.isRangeUpdated:
				nowTime = rospy.Time.now()
				self.dt = (nowTime.secs - self.lastTime.secs)+(nowTime.nsecs - self.lastTime.nsecs)/1000000000.0
				# Update F matrix
				o = self.yawMeas #orientation



				self.filter.F = np.array([  [1., 0.,   math.sin(o)*self.dt],
											[0., 1.,   math.cos(o)*self.dt],
											[0., 0.,                    1.] ]) 

				# Predite state

				self.filter.predict()

				# Correct

				self.measurement = [[self.rangeMeas]]
				self.filter.update(self.measurement , H_Jac_31_eh01  , H_meas_31_eh01, args = self.anchorPos, hx_args = self.anchorPos)
				self.state = self.filter.x
				self.bdFrame()
				#print(" State now ", self.state)

				#self.isImuUpdated = False
				self.isRangeUpdated = False
				self.isYawUpdated = False
				self.lastTime = nowTime

			pass
		pass


	def imuUpdate(self, msg):


		delt = (msg.header.stamp.secs-self.lastImu.header.stamp.secs)+(msg.header.stamp.nsecs-self.lastImu.header.stamp.nsecs)/1000000000.0

		orientation_list = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
		(yaw, roll, pitch) = tf.transformations.euler_from_quaternion(orientation_list)
		self.yawMeas = yaw



		if self.isImuUpdated == True:
			self.vtest += delt*(msg.linear_acceleration.y+0.65)
			#print(" Velocity ", self.vtest, " delt ", delt, " vel ", msg.linear_acceleration.y)
			vel = Float64(self.vtest)
			self.pub_vel.publish(vel)
			pass

		self.lastImu = msg
		self.isImuUpdated  = True
		pass



	def rangeUpdate(self, msg):
		self.isRangeUpdated = True
		self.rangeMeas = 0.9804*msg.data/1000. - 0.5882
		print("range meas ", self.rangeMeas)
		pass

	def yawUpdate(self, msg):
		self.isYawUpdated = True
		self.yawMeas = msg.vector.z
		print("imu meas ", self.yawMeas)
		pass


	def bdFrame(self):
		br = tf.TransformBroadcaster()
		br.sendTransform((self.state[0], self.state[1], 0), tf.transformations.quaternion_from_euler(0, 0, self.yawMeas), rospy.Time.now(), "map", "vanilarEKF")
		pass



if __name__ == '__main__':
	rospy.init_node('vanilarEKF', anonymous=True)
	vtracker = Tracker()
	vtracker.efkInitialization()

	rospy.loginfo("Start the vanilarEKF Tracker")
	#rate = rospy.Rate(100) # 10hz

	vtracker.ekfLoop()  	