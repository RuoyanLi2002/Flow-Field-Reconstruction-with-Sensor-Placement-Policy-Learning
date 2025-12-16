import os
import math
import pickle
import netCDF4 as nc
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader



class MeanVarianceScaler:
    def __init__(self, path):
        with open(path, "rb") as f:
            stats_dict = pickle.load(f)

        self.mean_ = torch.from_numpy(stats_dict["mean"])
        self.var_ = torch.from_numpy(stats_dict["variance"])
        self.std_ = torch.sqrt(torch.from_numpy(stats_dict["variance"]))

        print(f"self.mean_: {self.mean_}")
        print(f"self.var_: {self.var_}")
        print(f"self.std_: {self.std_}")

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        self.mean_ = self.mean_.to(X.device)
        self.std_ = self.std_.to(X.device)

        X_scaled = (X - self.mean_) / self.std_
        return X_scaled

    def inverse_transform(self, X_scaled):
        if not isinstance(X_scaled, torch.Tensor):
            X_scaled = torch.tensor(X_scaled, dtype=torch.float32)

        self.mean_ = self.mean_.to(X_scaled.device)
        self.std_ = self.std_.to(X_scaled.device)

        X_original = X_scaled * self.std_ + self.mean_
        return X_original
    

def create_edge_index_and_attributes(coords, k=6):
    """
    Args:
        coords (Tensor): Tensor of shape [num_points, 3] representing 3D coordinates.
        k (int): Number of nearest neighbors to connect.

    Returns:
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        edge_attr (Tensor): Edge attributes tensor of shape [num_edges, 4], where the
                            first three columns are the 3D vectors and the fourth column is the norm.
    """
    coords = coords.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    edges = []
    vectors = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                edges.append((i, j))
                vector = coords[j] - coords[i]
                vectors.append(vector)

    edge_index = torch.tensor(edges, dtype=torch.long).t()   # Shape [2, num_edges]
    vectors = torch.tensor(vectors, dtype=torch.float32)     # Shape [num_edges, 3]
    norms = torch.norm(vectors, dim=1, keepdim=True)         # Shape [num_edges, 1]

    edge_attr = torch.cat([vectors, norms], dim=1)           # Shape [num_edges, 4]

    return edge_index, edge_attr

def load_single_data(args, index, scaler, data_type):
    dataset = []

    for i in index:
        print(f"i: {i}")
        file_name = f'{args.data_name}{i}.pkl'
        input_file_path = os.path.join(args.data_path, file_name)

        with open(input_file_path, 'rb') as f:
            data_dict = pickle.load(f)

        locations = torch.from_numpy(data_dict['locations']) # Shape: (num_points, 3)
        data = torch.from_numpy(data_dict['data']) # Shape: (num_frames, num_points, 4)
        to_interpolate_mask = torch.from_numpy(data_dict['to_interpolate']) # Shape: (num_points, )

        to_interpolate_mask = to_interpolate_mask.unsqueeze(-1)
        edge_index, edge_attr = create_edge_index_and_attributes(locations)

        mask_bool = to_interpolate_mask.squeeze(-1).bool()
        interpolate_indices = torch.where(mask_bool)[0]
        non_interpolate_indices = torch.where(~mask_bool)[0]

        for j in range(data.shape[0]):
            if args.scale:
                frame_data = scaler.transform(data[j])
            else:
                frame_data = data[j]

            if args.use_noise:
                sigma2 = 0.1
                noise  = torch.randn_like(frame_data) * sigma2**0.5
                frame_data = frame_data + noise

            interpolate_locations = locations[interpolate_indices]
            non_interpolate_locations = locations[non_interpolate_indices]
            distances = torch.cdist(interpolate_locations, non_interpolate_locations, p=2)
            nearest_neighbor_indices = distances.argmin(dim=1)
            nearest_indices = non_interpolate_indices[nearest_neighbor_indices]
            nearest_features = frame_data[nearest_indices]

            input_data = frame_data.clone()
            input_data[interpolate_indices] = nearest_features

            graph_data = Data(x=input_data, edge_index=edge_index, edge_attr=edge_attr, y=frame_data, to_interpolate_mask=to_interpolate_mask, pos=locations)
            dataset.append(graph_data)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if not os.path.exists(args.data_save_path):
        os.makedirs(args.data_save_path)

    torch.save(dataset, f"{args.data_save_path}/{data_type}.pth")

    return dataloader

def load_train(args, split="train"):
    file_path = f"{args.data_save_path}/{split}.pth"

    if False and os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        dataloader = DataLoader(all_data, batch_size=args.batch_size, shuffle=True)
    else:
        print(f"{split}.pth does not exists. Create dataset")

        scaler = MeanVarianceScaler(args.stats_path)
        split_dict = np.load(args.split_path, allow_pickle=True).item()
        index = split_dict[split]

        dataloader = load_single_data(args, index, scaler, data_type=split)

    return dataloader

def load_valid(args, split="valid"):
    file_path = f"{args.data_save_path}/{split}.pth"

    if os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        dataloader = DataLoader(all_data, batch_size=args.batch_size, shuffle=True)
    else:
        print(f"{split}.pth does not exists. Create dataset")

        scaler = MeanVarianceScaler(args.stats_path)
        split_dict = np.load(args.split_path, allow_pickle=True).item()
        index = split_dict[split]

        dataloader = load_single_data(args, index, scaler, data_type=split)

    return dataloader


def load_test(args, split="test"):
    file_path = f"{args.data_save_path}/{split}.pth"

    if os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        dataloader = DataLoader(all_data, batch_size=args.batch_size, shuffle=True)
    else:
        print(f"{split}.pth does not exists. Create dataset")

        scaler = MeanVarianceScaler(args.stats_path)
        split_dict = np.load(args.split_path, allow_pickle=True).item()
        index = split_dict[split]

        dataloader = load_single_data(args, index, scaler, data_type=split)

    return dataloader

def load_sst(args, split):
    dataset = nc.Dataset(args.data_path, 'r')

    lat = np.array(dataset.variables['lat'][:])
    lon = np.array(dataset.variables['lon'][:])
    sst = np.array(dataset.variables['sst'][:])
    time = np.array(dataset.variables['time'][:])
    time_bnds = np.array(dataset.variables['time_bnds'][:])

    print("Shape of lat:", lat.shape)
    print("Shape of lon:", lon.shape)
    print("Shape of sst:", sst.shape)
    print("Shape of time:", time.shape)
    print("Shape of time_bnds:", time_bnds.shape)
    
    if split == "train":
        sst = sst[:int(0.8 * sst.shape[0]), :, :]
    else:
        sst = sst[int(0.9 * sst.shape[0]):, :, :]
    
    print("Shape of sst:", sst.shape)

    print(f"np.mean(sst): {np.mean(sst)}")

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    grid_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    num_points = grid_points.shape[0]
    
    num_known = int(num_points * args.ratio)

    n_neighbors = 2
    loop = False

    lin1 = torch.linspace(0, 1, lat.shape[0])
    lin2 = torch.linspace(0, 2, lon.shape[0])
    xx, yy = torch.meshgrid(lin1, lin2, indexing="ij")
    pos = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    dx = lin1[1] - lin1[0]
    radius = n_neighbors * dx * math.sqrt(2) + 1e-5
    dist = torch.cdist(pos, pos)
    mask = dist < radius
    
    if not loop:
        mask.fill_diagonal_(False)
    
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()

    src, dst = edge_index
    edge_vector = grid_points[dst] - grid_points[src]
    edge_vector = torch.from_numpy(edge_vector)
    edge_norm = edge_vector.norm(dim=1, keepdim=True)
    edge_attr = torch.cat((edge_vector, edge_norm), dim=1)

    dataset = []
    for i in range(sst.shape[0]):
        known_indices = np.random.choice(num_points, size=num_known, replace=False)
    
        to_interpolate_mask = np.ones(num_points, dtype=int)
        to_interpolate_mask[known_indices] = 0
        to_interpolate_mask = to_interpolate_mask.reshape(-1, 1)

        sst_frame = sst[i, :, :]
        sst_flat = sst_frame.ravel()
        
        sst_missing = sst_flat.copy()
        sst_missing[to_interpolate_mask.ravel() == 1] = np.nan
        
        known_points = grid_points[known_indices]
        known_values = sst_flat[known_indices]

        sst_interpolated = griddata(
            known_points,
            known_values,
            grid_points,
            method='nearest'
        )
        
        x = torch.from_numpy(sst_interpolated.reshape(-1, 1))

        y = torch.from_numpy(sst_frame.ravel().reshape(-1, 1))
        to_interpolate_mask = torch.from_numpy(to_interpolate_mask)

        print(f"x: {x.shape} {type(x)}")
        print(f"edge_index: {edge_index.shape} {type(edge_index)}")
        print(f"edge_attr: {edge_attr.shape} {type(edge_attr)}")
        print(f"y: {y.shape} {type(y)}")
        print(f"to_interpolate_mask: {to_interpolate_mask.shape} {type(to_interpolate_mask)}")
        print(f"----------{i}----------")
        # exit()

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, to_interpolate_mask=to_interpolate_mask)
        dataset.append(graph_data)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataloader, lat, lon