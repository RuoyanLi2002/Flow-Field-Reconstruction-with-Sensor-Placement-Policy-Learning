import os
import pickle
import numpy as np

input_folder = "airfoil_pickles"

sum_data = np.zeros(4)
sum_squared_data = np.zeros(4)
total_points = 0

for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".pkl"):
        file_path = os.path.join(input_folder, filename)
        
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        data = data_dict['data']
        print(f"data: {data.shape}")

        data = data.reshape(-1, 4)
        
        if sum_data is None:
            sum_data = np.zeros_like(data[0])
            sum_squared_data = np.zeros_like(data[0])
        # print(f"np.sum(data, axis=0): {np.sum(data, axis=0).shape}")
        sum_data += np.sum(data, axis=0)
        sum_squared_data += np.sum(data**2, axis=0)
        total_points += data.shape[0]

        # print(f"np.sum(data, axis=0): {np.sum(data, axis=0).shape}")
        # print(f"np.sum(data**2, axis=0): {np.sum(data**2, axis=0).shape}")
        # print(f"data.shape[0]: {data.shape[0]}")
        # print("----------")

mean_data = sum_data / total_points
variance_data = (sum_squared_data / total_points) - (mean_data**2)

# Print the results
print("Element-wise mean of the feature vectors:")
print(mean_data)
print("\nElement-wise variance of the feature vectors:")
print(variance_data)

stats_dict = {"mean": mean_data, "variance": variance_data}

with open("sphere_stats.pkl", "wb") as f:
    pickle.dump(stats_dict, f)