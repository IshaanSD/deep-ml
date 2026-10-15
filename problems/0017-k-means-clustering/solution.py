import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	m,n = len(points), len(points[0])
	if max_iterations == 0: return initial_centroids
	def calc_dist(p1: tuple[float], p2: tuple[float]):
		return (((np.array(p1) - np.array(p2))**2).sum())**(1/2)
	
	centroids_sums = {centroid:np.zeros(n) for centroid in initial_centroids}
	centroids_nums = {centroid:0 for centroid in initial_centroids}
	for point in points:
		nearest_centroid_dist = (float('inf'), None)
		for centroid in initial_centroids:
			nearest_centroid_dist = min((calc_dist(point,centroid),centroid), nearest_centroid_dist)
		nearest_centroid = nearest_centroid_dist[1]
		centroids_sums[nearest_centroid] += np.array(point)
		centroids_nums[nearest_centroid] += 1

	centroid_means = np.array([centroids_sums[centroid] for centroid in centroids_sums])
	centroid_means /= np.array([centroids_nums[centroid] for centroid in centroids_nums]).reshape(-1,1)
	centroid_means = centroid_means.round(2)
	
	centroid_means = [tuple(centroid_mean) for centroid_mean in centroid_means]
	final_centroids =  k_means_clustering(points, k, centroid_means, max_iterations-1)

	return final_centroids