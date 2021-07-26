import trajGenerator as trg
from interpolateTraj import interpolateTraj
import robosuite
from robosuite.models.objects import BoxObject
from robosuite.models.robots import Wombat_arm
from robosuite.models.arenas import EmptyArena
from robosuite.models.grippers import gripper_factory
import invK
import math
from mujoco_py import MjSim, MjViewer
import numpy as np
import time
import matplotlib as mpl
import ipdb
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
from robosuite.models import MujocoWorldBase

class D3_pick_place_env(object):

	def __init__ (self,is_render):

		self.action_dim = 6
		self.obs_dim = 19
		self.q_pos_last = np.zeros(self.action_dim)
		self.observation_current = np.zeros(self.obs_dim)
		self.observation_last = np.zeros(self.obs_dim)
		self.observation_last2last = np.zeros(self.obs_dim)
		self.is_render = True
		self.done = False
		pass

	def set_env(self):

		self.world = MujocoWorldBase()
		self.mujoco_robot = Wombat_arm()

		self.mujoco_robot.set_base_xpos([0, 0.0, 0])
		self.world.merge(self.mujoco_robot)

		self.mujoco_arena =EmptyArena()
		# mujoco_arena.set_origin([0.8, 0, 0])
		self.world.merge(self.mujoco_arena)

		self.iphonebox = BoxObject(name="iphonebox",size=[0.055,0.11,0.0069],rgba=[0,0,0,1],friction=[1,1,1]).get_obj()
		self.iphonebox.set('pos', '0.6 -2 1')
		self.world.worldbody.append(self.iphonebox)

		self.box = BoxObject(name="box",size=[0.35,9.7,0.37],rgba=[0.5,0.5,0.5,1],friction=[1,1,1]).get_obj()
		self.box.set('pos', '0.6 -2 0')
		self.world.worldbody.append(self.box)

		self.model = self.world.get_model(mode="mujoco_py")

		self.sim = MjSim(self.model)
		
		if self.is_render:
			self.viewer = MjViewer(self.sim)
			self.viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh
			self.viewer.render()

		self.timestep= 0.0005
		self.sim_state = self.sim.get_state()

	def get_signal(self,action,obs_last,obs_last2last):



		action = action.reshape(1,-1)
		# ipdb.set_trace()
		joint_target_action=trg.jointTraj(action)
		# q_pos_last[0:6] = obs_last2last[0:6] 
		joint_real=joint_target_action
		ee_pose=invK.real2sim_wrapper(action[0])
		joint_sim=invK.ik_wrapper(joint_real[0])

		PD_scale= self.PD_signal_scale(action[0],joint_real[0])

		PD_signal=[self.PD_controller_rot(joint_sim[3],obs_last[0],obs_last2last[0],PD_scale[0]),
			   self.PD_controller_rot(joint_sim[4],obs_last[1],obs_last2last[1],PD_scale[1]),
			   self.PD_controller_rot(joint_sim[5],obs_last[2],obs_last2last[2],PD_scale[2]),
			   self.PD_controller_lin(joint_sim[0],obs_last[3],obs_last2last[3],PD_scale[3]),
			   self.PD_controller_lin(joint_sim[1],obs_last[4],obs_last2last[4],PD_scale[4]),
			   self.PD_controller_lin(joint_sim[2],obs_last[5],obs_last2last[5],PD_scale[5])]


		return PD_signal


	def step(self,action):
		# action = action.reshape(1,-1)
		self.observation_last2last = self.observation_last
		self.observation_last = self.observation_current
		PD_signal = self.get_signal(action,self.observation_last,self.observation_last2last)
		self.sim.data.ctrl[0:6] = PD_signal[0:6]
		self.sim.step()
		self.viewer.render()
		# print("sending steps")
		self.observation_current = self.get_obseravtion()
		self.reward = self.calculate_reward(self.observation_current)
		self.sim.data.set_joint_qvel('box_joint0', [0, 0.4, 0, 0, 0, 0])
		self.done = self.is_done
		return self.observation_current,self.reward,self.done,None


	def get_obseravtion(self):

		observation = np.zeros(self.obs_dim)

		observation[0] = self.sim.data.get_joint_qpos("robot0_branch1_joint")
		observation[1] = self.sim.data.get_joint_qpos("robot0_branch2_joint")
		observation[2] = self.sim.data.get_joint_qpos("robot0_branch3_joint")
		#linear
		observation[3] = self.sim.data.get_joint_qpos("robot0_branch1_linear_joint")
		observation[4] = self.sim.data.get_joint_qpos("robot0_branch2_linear_joint")
		observation[5] = self.sim.data.get_joint_qpos("robot0_branch3_linear_joint")

		observation[6] = self.sim.data.get_joint_qvel("robot0_branch1_joint")
		observation[7] = self.sim.data.get_joint_qvel("robot0_branch2_joint")
		observation[8] = self.sim.data.get_joint_qvel("robot0_branch3_joint")
		#linear
		observation[9] = self.sim.data.get_joint_qvel("robot0_branch1_linear_joint")
		observation[10] = self.sim.data.get_joint_qvel("robot0_branch2_linear_joint")
		observation[11] = self.sim.data.get_joint_qvel("robot0_branch3_linear_joint")


		observation[12:19] = self.sim.data.get_joint_qpos('iphonebox_joint0')

		return observation

		pass


	def calculate_reward(self,obs):

		return 10


	def is_done(self,obs):

		return False

	def PD_controller_rot(self,des,current,q_pos_last,scale):
	
		kp=20*scale 
		kd=0.6
		qpos = des+kp*(des-current)-kd*(current-q_pos_last)
		# print(kp*(des-current))
		return qpos

	def PD_controller_lin(self,des,current,q_pos_last,scale):
		
		kp=150
		kd=1500
		qpos = des+kp*(des-current)-kd*(current-q_pos_last)
		# print(kp*(des-current))
		return qpos

	def PD_signal_scale(self,ee_pos,joint_vals):

		#scales the PD signal based on the ee pos or joint values; wombat_arm needs
		#different PD values depending on where it is, position-wise

		ee_xy_disp=np.array([math.sqrt(ee_pos[0]**2+ee_pos[1]**2)]*6)+1.0
		lin_vals=np.array([joint_vals[2],joint_vals[0],joint_vals[1]]*2)+1.0
		scale=7
		PD_scale_factor=((np.multiply(ee_xy_disp,lin_vals)**2)-1)*scale
		return PD_scale_factor