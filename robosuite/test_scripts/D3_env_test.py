import numpy as np

from D3_pick_place import D3_pick_place_env

if __name__ == '__main__':

	timestep = 10000
	t = 0

	D3_pp = D3_pick_place_env(True)
	D3_pp.set_env()
	action_zero = np.array([0,0,0.6,0,0,0])
	while t<timestep:

		if t>0 and t<5000:
			action = action_zero
		else:
			action = np.array([0,0,0.9,0,0,0])
		# action[2] += t*0.00001

		D3_pp.step(action)
		print(t,action)
		t += 1
