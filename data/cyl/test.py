import os
import pickle
import meshio
import numpy as np
from decimal import Decimal

folder = "raw_data"
output_folder = "cyl_pickles"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(folder):
    print(f"Found {filename}")
    tmp_filename = filename.split(".")[0]
    print(f"tmp_filename: {tmp_filename}")
    vtu_file = os.path.join(folder, filename)
    
    mesh = meshio.read(vtu_file)
    # print(mesh)
    # exit()
    locations = mesh.points

    num_points = locations.shape[0]
    num_timesteps = 501
    data_array = np.zeros((num_timesteps, num_points, 4))

    for i in range(num_timesteps):
        t = i * 0.01
        t = str(Decimal(t).quantize(Decimal("1.00")).normalize())
        
        vx_name = f"Velocity_field,_x-component_@_t={t}"
        vy_name = f"Velocity_field,_y-component_@_t={t}"
        vz_name = f"Velocity_field,_z-component_@_t={t}"
        pressure_name = f"Pressure_@_t={t}"

        try:
            vx = mesh.point_data[vx_name]
            vy = mesh.point_data[vy_name]
            vz = mesh.point_data[vz_name]
            pressure = mesh.point_data[pressure_name]
        except KeyError as e:
            print(f"Data missing for {vtu_file} at t={t}: {e}")
            print(f"t: {t}")
            print(f"e: {e}")
            exit()
            continue

        data_array[i, :, 0] = vx
        data_array[i, :, 1] = vy
        data_array[i, :, 2] = vz
        data_array[i, :, 3] = pressure


    output_file = os.path.join(output_folder, f"{tmp_filename}.pkl")
    data_dict = {"locations": locations, "data": data_array}
    with open(output_file, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"Processed and saved: {vtu_file} -> {output_file}")
