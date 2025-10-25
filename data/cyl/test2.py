import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

input_folder = 'cyl_pickles'
output_folder = 'cyl_uniform30'

os.makedirs(output_folder, exist_ok=True)

count = 0
for filename in os.listdir(input_folder):
    print(f"Found {filename}")
    input_file_path = os.path.join(input_folder, filename)
    
    with open(input_file_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    locations = data_dict['locations']  # Shape: (num_points, 3)
    data = data_dict['data']            # Shape: (num_points, 4)
    num_points = locations.shape[0]
    
    num_known = int(np.ceil(0.30 * num_points))
    
    kmeans = KMeans(n_clusters=num_known, init='k-means++', random_state=42)
    kmeans.fit(locations)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # For each cluster, find the point closest to the centroid
    known_indices = []
    for cluster_id in range(num_known):
        # Get indices of points in the current cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = locations[cluster_indices]
        
        # Compute distances from cluster points to the cluster centroid
        centroid = cluster_centers[cluster_id].reshape(1, -1)
        distances = cdist(cluster_points, centroid, metric='euclidean').flatten()
        
        # Find the index of the point closest to the centroid
        closest_point_idx = cluster_indices[np.argmin(distances)]
        # print(f"cluster_id: {cluster_id}, closest_point_idx: {closest_point_idx}")
        known_indices.append(closest_point_idx)
    
    # Initialize 'to_interpolate' mask
    to_interpolate = np.ones(num_points, dtype=int)  # Start with all points unknown
    to_interpolate[known_indices] = 0                # Mark known points
    # print(f"num_points: {num_points}, num_known: {num_known}, np.sum(to_interpolate): {np.sum(to_interpolate)}")
    # Add 'to_interpolate' to the dictionary
    data_dict['to_interpolate'] = to_interpolate
    # exit()
    # Save the modified dictionary to the output folder
    output_file_path = os.path.join(output_folder, f"cyl{count}.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump(data_dict, f)
    count += 1
    
    print(f"Processed {filename} and saved to {output_file_path}")
