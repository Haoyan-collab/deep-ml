import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	def cal_distance(point, centroids):
		distances = np.sqrt(np.sum((point-centroids)**2, axis=-1))
		return np.argmin(distances, axis=-1)

	points = np.array(points)
	centroids = np.array(initial_centroids,dtype = float)
	for _ in range(max_iterations):
		clusters = [[] for _ in range(k)]
		new_centroids = []
		for point in points:
			idx = cal_distance(point,centroids)
			clusters[idx].append(point)
		for i,cluster in enumerate(clusters):
			
			if len(cluster)==0:
				new_centroids.append(centroids[i])
			else:
				new_centroids.append(np.mean(cluster, axis=0))
		new_centroids = np.array(new_centroids)
		if np.allclose(new_centroids,centroids):
			break
		centroids = new_centroids
	final_centroids = [tuple(np.round(point, 4)) for point in centroids]
	return final_centroids