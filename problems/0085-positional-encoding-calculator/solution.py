import numpy as np

def pos_encoding(position: int, d_model: int):
	# Your code here
	if position == 0 or d_model <= 0:
		return -1

	# pos(position, 1), to later calculate with (dims,)
	pos = np.arange(position).reshape(position,1)
	dims = np.arange(0, d_model, 2)
	angels = pos / (10000**(dims/d_model)) #(position,dims)

	pe = np.zeros((position,d_model))
	pe[:,0:d_model:2] = np.sin(angels)
	#if d_model is odd, 1:d_model:2 != d_model / 2
	pe[:,1:d_model:2] = np.cos(angels[:,:pe[:,1:d_model:2].shape[-1]]) 

	pos_encoding = np.float16(pe)
	return pos_encoding