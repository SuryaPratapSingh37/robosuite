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

##PD controller below
def PD_controller_rot(des,current,q_pos_last,scale):
	
	#kp = 10
	#kp=10
	#kp = 1
	#kd = 0.3
	#kp = 5
	#kd = 0.6
	kp=20*scale
	kd=0.6
	qpos = des+kp*(des-current)-kd*(current-q_pos_last)
	# print(kp*(des-current))
	return qpos

	# return np.array(points)
def PD_controller_lin(des,current,q_pos_last,scale):
	
	#kp = 10
	#kd = 0.8
	#kp=10
	#kd=0.1
	kp=150
	kd=1500
	qpos = des+kp*(des-current)-kd*(current-q_pos_last)
	# print(kp*(des-current))
	return qpos

#scales the PD signal based on the ee pos or joint values; wombat_arm needs
#different PD values depending on where it is, position-wise
def PD_signal_scale(ee_pos,joint_vals):
	ee_xy_disp=np.array([math.sqrt(ee_pos[0]**2+ee_pos[1]**2)]*6)+1.0
	lin_vals=np.array([joint_vals[2],joint_vals[0],joint_vals[1]]*2)+1.0
	scale=7
	PD_scale_factor=((np.multiply(ee_xy_disp,lin_vals)**2)-1)*scale
	#print("PD_scale_factor:",PD_scale_factor)
	#PD_scale_factor=np.array([1,1,1,1,1,1])
	return PD_scale_factor

##Building a custom world, importing wombat_arm and adding gripper below
world = MujocoWorldBase()
mujoco_robot = Wombat_arm()

# gripper = gripper_factory(None)
gripper = gripper_factory('D3_gripper')
#gripper.hide_visualization()
mujoco_robot.add_gripper(gripper)

# mujoco_robot.set_base_xpos([0.4, 0.06, 0])
mujoco_robot.set_base_xpos([0, 0.0, 0])
world.merge(mujoco_robot)

#mujoco_arena = TableArena()
# mujoco_arena = BinsArena()
mujoco_arena =EmptyArena()
# mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

##adding iphone as a box and conveyor belt as a box below
#sphere = BallObject(
#    name="sphere",
#    size=[0.04],
#    rgba=[0, 0.5, 0.5, 1])#.get_collision()
#sphere.append(new_joint(name='sphere_free_joint', type='free'))
#sphere.set('pos', '1.0 0 1.0')
#world.worldbody.append(sphere)
# iphonebox = BoxObject(name="iphonebox",size=[0.04,0.04,0.02],rgba=[0,0.5,0.5,1],friction=[1,1,1]).get_obj()
# iphonebox.set('pos', '0.6 -2 1')
# world.worldbody.append(iphonebox)
iphonebox = BoxObject(name="iphonebox",size=[0.055,0.11,0.0069],rgba=[0,0,0,1],friction=[1,1,1]).get_obj()
iphonebox.set('pos', '0.6 -2 1')
world.worldbody.append(iphonebox)

box = BoxObject(name="box",size=[0.35,9.7,0.37],rgba=[0.5,0.5,0.5,1],friction=[1,1,1]).get_obj()
box.set('pos', '0.6 -2 0')
world.worldbody.append(box)

model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

viewer.render()

t_final = 10000
torque1=[[None]]*t_final
torque2=[[None]]*t_final
torque3=[[None]]*t_final
torque4=[[None]]*t_final
torque5=[[None]]*t_final
torque6=[[None]]*t_final
t = 0
timestep= 0.0005

sim_state = sim.get_state()
q_pos_last = [0]*6
ee_pose_des = []
ee_pose_current = []
t_arr=np.linspace(timestep*t,timestep*t_final,t_final-t)


#horizontal circle
r=0.1
tLin=2000
dt=1.0/(t_final-t-tLin)
target_traj1=np.array([[0,-r*np.sin(np.pi/2*(float(i)/tLin)),0.6,0,0,0] for i in range(0,tLin)])
target_traj2=np.array([[-r*np.sin((i)*dt*np.pi*2),-r*np.cos(i*dt*np.pi*2),0.6,0,0,0] for i in range(t,t_final-tLin)])
target_traj=np.block([[target_traj1],[target_traj2]])
#vertical circle
# r=0.1
# dt=1.0/(t_final-t)
# target_traj=np.array([[r*np.sin(i*dt*np.pi*2),0.0,0.7-r*np.cos(i*dt*np.pi*2),0,0,0] for i in range(t,t_final)])
# target_traj=np.array([[-0.2,0,0.7,0,0,0] for i in range(t,t_final)])
#convert to joint trajectory
joint_target_traj=trg.jointTraj(target_traj)
sim.set_state(sim_state)
ee_pose_des = []

#joint values in the simulation
j_actual=np.zeros((t_final-t,6))
j_actual_real=np.zeros((t_final-t,6))
#goal joint values for simulation to follow
j_goal=np.zeros((t_final-t,6))

while t<t_final:
	#rotary
	q_pos_last[0] = sim.data.get_joint_qpos("robot0_branch1_joint")
	q_pos_last[1] = sim.data.get_joint_qpos("robot0_branch2_joint")
	q_pos_last[2] = sim.data.get_joint_qpos("robot0_branch3_joint")
	#linear
	q_pos_last[3] = sim.data.get_joint_qpos("robot0_branch1_linear_joint")
	q_pos_last[4] = sim.data.get_joint_qpos("robot0_branch2_linear_joint")
	q_pos_last[5] = sim.data.get_joint_qpos("robot0_branch3_linear_joint")
	# print("q_pos_last",q_pos_last)
	sim.step()
	#print(sim.data.get_joint_qpos("branch1_joint"),sim.data.get_joint_qpos("branch2_joint"),sim.data.get_joint_qpos("branch3_joint"),sim.data.get_joint_qpos("branch1_linear_joint"),sim.data.get_joint_qpos("branch2_linear_joint"),sim.data.get_joint_qpos("branch3_linear_joint"))
	if True:
		viewer.render()

	#current target joint values, in IK frame
	joint_real=joint_target_traj[0]
	ee_pose=invK.real2sim_wrapper(target_traj[0])
	# ee_pose=[0.2,0,0.7,0,0,0]
	# print("target in real frame",target_traj[t])
	# print("target in sim frame",ee_pose)
	ee_pose_des.append(ee_pose)
	#convert current target joint values, in sim frame
	joint_sim=invK.ik_wrapper(joint_real)
	# print("joint_real",joint_real)
	# print("joint_sim",joint_sim)
	j_goal[t,:]=np.array(joint_sim)
	#calculate/send PD control signal to the motors
	# q_pos_last = np.array([sim.data.get_joint_qpos("robot0_branch1_joint"),sim.data.get_joint_qpos("robot0_branch2_joint"),sim.data.get_joint_qpos("robot0_branch3_joint"),sim.data.get_joint_qpos("robot0_branch1_linear_joint"),sim.data.get_joint_qpos("robot0_branch2_linear_joint"),sim.data.get_joint_qpos("robot0_branch3_linear_joint")])
	PD_scale=PD_signal_scale(target_traj[0],joint_target_traj[0])
	# print("PD_scale",PD_scale)
	PD_signal=[PD_controller_rot(joint_sim[3],sim.data.get_joint_qpos("robot0_branch1_joint"),q_pos_last[0],PD_scale[0]),
			   PD_controller_rot(joint_sim[4],sim.data.get_joint_qpos("robot0_branch2_joint"),q_pos_last[1],PD_scale[1]),
			   PD_controller_rot(joint_sim[5],sim.data.get_joint_qpos("robot0_branch3_joint"),q_pos_last[2],PD_scale[2]),
			   PD_controller_lin(joint_sim[0],sim.data.get_joint_qpos("robot0_branch1_linear_joint"),q_pos_last[3],PD_scale[3]),
			   PD_controller_lin(joint_sim[1],sim.data.get_joint_qpos("robot0_branch2_linear_joint"),q_pos_last[4],PD_scale[4]),
			   PD_controller_lin(joint_sim[2],sim.data.get_joint_qpos("robot0_branch3_linear_joint"),q_pos_last[5],PD_scale[5])]
	# print("PD_signal",PD_signal)
	
	sim.data.ctrl[0]=PD_signal[0]
	torque1[t]=PD_scale[0]
	sim.data.ctrl[1]=PD_signal[1]
	torque2[t]=PD_scale[1]
	sim.data.ctrl[2]=PD_signal[2]
	torque3[t]=PD_scale[2]
	sim.data.ctrl[3]=PD_signal[3]
	torque4[t]=PD_scale[3]
	sim.data.ctrl[4]=PD_signal[4]
	torque5[t]=PD_scale[4]
	sim.data.ctrl[5]=PD_signal[5]
	torque6[t]=PD_scale[5]
	sim.data.set_joint_qvel('box_joint0', [0, 0.4, 0, 0, 0, 0])
	##iphonebox pose
	print("iphonebox_pose: ",sim.data.get_joint_qpos('iphonebox_joint0'))
	##setting gripper fingers values
	sim.data.set_joint_qpos('gripper0_left_finger_joint', -0.18)
	sim.data.set_joint_qpos('gripper0_right_finger_joint',-0.18)
	##printing gripper fingers velocities
	print(sim.data.get_joint_qvel('gripper0_left_finger_joint'))
	print(sim.data.get_joint_qvel('gripper0_right_finger_joint'))
	"""Available "joint" names = ('robot0_branch1_joint', 'robot0_branch1_pivot_joint', 'robot0_branch1_linear_joint', 
	'robot0_branch1_linear_revolute_joint', 'robot0_branch1_clevis_joint', 'robot0_branch1_ee_joint', 'gripper0_left_finger_joint',
	 'gripper0_right_finger_joint', 'robot0_branch2_joint', 'robot0_branch2_pivot_joint', 'robot0_branch2_linear_joint', 
	 'robot0_branch2_linear_revolute_joint', 'robot0_branch2_clevis_joint', 'robot0_branch2_ee_joint', 'robot0_branch3_joint',
	  'robot0_branch3_pivot_joint', 'robot0_branch3_linear_joint', 'robot0_branch3_linear_revolute_joint', 
	  'robot0_branch3_clevis_joint', 'robot0_branch3_ee_joint', 'iphonebox_joint0', 'box_joint0')"""
	##Just like the code between lines 195-203 you can either set the 'joint pos' and 'joint_vel' or you can get/obtain the 'joint_pos' and 'joint_vel'

	
	t=t+1