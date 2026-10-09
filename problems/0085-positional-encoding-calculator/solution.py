import numpy as np

def pos_encoding(position: int, d_model: int):
	# Your code here
	if position == 0 or d_model <= 0:
		return -1

	i = np.arange(0, d_model, 2)  #(d_model/2,)
	pos = np.arange(position)[:,None] #(position, 1)

	angles = pos / (10000**(i / d_model)) #(position, d_model/2)

	pe = np.zeros((position, d_model))
	pe[:,0::2] = np.sin(angles)
	pe[:,1::2] = np.cos(angles[:,:pe[:,1::2].shape[-1]])

	return pe.astype(np.float16)
