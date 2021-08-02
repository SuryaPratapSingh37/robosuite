import numpy as np
import ipdb
import matplotlib.pyplot as plt

from D3_pick_place import D3_pick_place_env

def plot_variables(var_readings,is_show = True):
	
	fig,b = plt.subplots(2,3)
	b[0][0].plot(var_readings[0],label="lin1 force_cmd")
	b[0][0].legend(loc='upper right')
	b[0][1].plot(var_readings[1],label="lin2 force_cmd")
	b[0][1].legend(loc='upper right')
	b[0][2].plot(var_readings[2],label="lin3 force_cmd")
	b[0][2].legend(loc='upper right')
	b[1][0].plot(var_readings[3],label="rot1 torque_cmd")
	b[1][0].legend(loc='upper right')
	b[1][1].plot(var_readings[4],label="rot2 torque_cmd")
	b[1][1].legend(loc='upper right')
	b[1][2].plot(var_readings[5],label="rot3 torque_cmd")
	b[1][2].legend(loc='upper right')


if __name__ == '__main__':

	timestep = 10000
	t = 0


	D3_pp = D3_pick_place_env(True)
	D3_pp.set_env()
	action_zero = np.array([0,0,0.6,0,0,0,-0.2,-0.2])
	obs_current = np.zeros(19)
	obs_last = np.zeros(19)
	gripper_to_finger = 0.09

	torque_reading = [[],[],[],[],[],[],[]]
	desired_reading = [[],[],[],[],[],[],[]]
	actual_reading = [[],[],[],[],[],[],[]]
	pos_error_reading = [[],[],[],[],[],[],[]]

	while t<timestep:

		##if pick and place

		if t>=0 and t<1000:


			obs_current,reward,done,_ = D3_pp.step(action_zero) 

			vel_iPhone = (obs_current[13] - obs_last[13])/(0.002/1.2)
			print("vel_iPhone: ",vel_iPhone)
			steps_to_reach = ((obs_current[20] - obs_current[13])/vel_iPhone)/(0.002/1.2)
			print("time_to_reach: ",steps_to_reach,t)
		
		elif t == 1000:
			vel_z = (obs_current[21] - 0.81)/int(steps_to_reach+250)
			print("vel_z: ",vel_z,steps_to_reach)
		else:
			if np.linalg.norm(obs_current[21]-0.81) > 0.035:
				pos_z = 0.6+vel_z*(t-1000)
				obs_current,reward,done,_ = D3_pp.step(np.array([0,0,pos_z,0,0,0,-0.2,-0.2])) 
				print("Phase 1: vel_z: ",np.linalg.norm(obs_current[21]-0.81))
			elif np.linalg.norm(obs_current[21]-0.81) > 0.0001 and np.linalg.norm(obs_current[21]-0.81)<0.035:
				pos_z = 0.6+vel_z*(t-1000)
				obs_current,reward,done,_ = D3_pp.step(np.array([0,0,pos_z,0,0,0,0.1,0.1])) 
				print("Phase 2: vel_z: ",np.linalg.norm(obs_current[21]-0.81))
				t_up = t
			else:
				D3_pp.step(np.array([0,0,pos_z-1e-4*(t-t_up),0,0,0,0.1,0.1])) 
				print("Phase 3: vel_z: ",pos_z-1e-4*(t-t_up))


		##if move sideways and test

		# action = np.array([0.0,0,0.7,0,0,0])
		# if t>=0 :
		# 	# action[1] -= t*0.00005
		# 	action[0] += t*0.00005
		# 	# action = action_zero
		# else:
		# 	pass

		# D3_pp.step(action)

		# obs_last = obs_current
		# # print(t,action)
		t += 1
		for t_n in range(6):
			torque_reading[t_n].append(D3_pp.sim.data.actuator_force[t_n])
			desired_reading[t_n].append(D3_pp.joint_sim[5-t_n])
			actual_reading[t_n].append(D3_pp.sim.data.get_joint_qpos(D3_pp.joint_names[t_n]))
			pos_error_reading[t_n].append(D3_pp.joint_sim[5-t_n]-D3_pp.sim.data.get_joint_qpos(D3_pp.joint_names[t_n]))
			# print("D3_pp.joint_names[t_n]",str(D3_pp.joint_names[t_n]))

	plot_variables(torque_reading)
	plot_variables(desired_reading)
	plot_variables(actual_reading)
	plot_variables(pos_error_reading)

	plt.show()


