
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	cnt = Counter(tuple(pair) for pair in data)
	tp = cnt[(1,1)]
	fn = cnt[(1,0)]
	fp = cnt[(0,1)]
	tn = cnt[(0,0)]
	return [[tp,fn],[fp,tn]]